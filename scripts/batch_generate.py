"""
batch_generate.py
-----------------
Queues multiple image generation jobs to a running ComfyUI instance via
its local REST API. Reads prompts from a JSON prompt library or a plain
text file (one prompt per line) and sends each one as a separate queue
entry using the specified workflow JSON.

Usage:
    # Use the structured JSON prompt library (recommended)
    .venv\\Scripts\\python.exe scripts\\batch_generate.py --mode surgeon
    .venv\\Scripts\\python.exe scripts\\batch_generate.py --mode clinic
    .venv\\Scripts\\python.exe scripts\\batch_generate.py --mode all
    .venv\\Scripts\\python.exe scripts\\batch_generate.py --mode all --count 3 --seed 42

    # Use a plain text prompts file (one prompt per line)
    .venv\\Scripts\\python.exe scripts\\batch_generate.py \\
        --workflow workflows\\Helios_Surgeon_v1.json \\
        --prompts training\\sample_prompts.txt

Options:
    --mode      Prompt mode: surgeon | clinic | all (uses prompts/ JSON library)
    --workflow  Path to a ComfyUI workflow JSON (overrides mode default)
    --prompts   Path to a plain-text prompts file (one per line)
    --host      ComfyUI host (default: 127.0.0.1:8188)
    --count     Number of times to generate each prompt (default: 1)
    --seed      Fixed random seed for reproducibility (default: random)
    --id        Run only the prompt with this specific ID (e.g. SURG_003)
"""

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

import requests
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
WORKFLOWS_DIR = ROOT / "workflows"
SURGEON_PROMPTS = PROMPTS_DIR / "surgeon_prompts.json"
CLINIC_PROMPTS = PROMPTS_DIR / "clinic_prompts.json"
VIDEO_PROMPTS = PROMPTS_DIR / "animatediff_prompts.json"
SURGEON_WORKFLOW = WORKFLOWS_DIR / "Helios_Surgeon_v1.json"
CLINIC_WORKFLOW = WORKFLOWS_DIR / "Helios_Clinic_v1.json"
VIDEO_WORKFLOW = WORKFLOWS_DIR / "Helios_AnimateDiff_Txt2Vid_v1.json"

POSITIVE_NODE_TITLE = "Positive Prompt"  # Must match node title in workflow JSON
NEGATIVE_NODE_TITLE = "Negative Prompt"  # Must match node title in workflow JSON
POLL_INTERVAL = 3        # seconds between queue status checks
MAX_WAIT_SECONDS = 300   # bail out after 5 minutes per job (prevents infinite hang)


# ── Loader helpers ───────────────────────────────────────────────────────────

def load_workflow(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_prompts(path: Path) -> list[dict]:
    """Load structured prompts from a JSON prompt library file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_prompts(path: Path) -> list[dict]:
    """Load plain-text prompts (one per line) and wrap in the standard dict schema."""
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    return [{"id": f"LINE_{i:03d}", "label": p[:40], "positive": p,
             "negative": "", "cfg": 7.0, "steps": 30} for i, p in enumerate(prompts)]


# ── Workflow injection helpers ────────────────────────────────────────────────

def find_node_by_title(workflow: dict, title: str) -> str | None:
    """Find a node's ID by its _meta title field."""
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            if node.get("_meta", {}).get("title", "").lower() == title.lower():
                return node_id
    return None


def inject_prompt(workflow: dict, prompt: dict, seed: int | None = None) -> dict:
    """Deep-copy the workflow and inject prompt text, cfg, steps, and seed."""
    wf = json.loads(json.dumps(workflow))  # deep copy

    # Positive prompt
    pos_id = find_node_by_title(wf, POSITIVE_NODE_TITLE)
    if pos_id:
        wf[pos_id]["inputs"]["text"] = prompt.get("positive", "")

    # Negative prompt
    neg_id = find_node_by_title(wf, NEGATIVE_NODE_TITLE)
    if neg_id:
        wf[neg_id]["inputs"]["text"] = prompt.get("negative", "")

    # KSampler — cfg, steps, seed
    for node in wf.values():
        if isinstance(node, dict) and node.get("class_type") == "KSampler":
            if "cfg" in prompt:
                node["inputs"]["cfg"] = prompt["cfg"]
            if "steps" in prompt:
                node["inputs"]["steps"] = prompt["steps"]
            if "sampler" in prompt:
                node["inputs"]["sampler_name"] = prompt["sampler"]
            if "scheduler" in prompt:
                node["inputs"]["scheduler"] = prompt["scheduler"]
            node["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)

    return wf


# ── ComfyUI API helpers ──────────────────────────────────────────────────────

def queue_prompt(workflow: dict, host: str) -> str:
    """POST the workflow to ComfyUI and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"http://{host}/prompt", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["prompt_id"]


def wait_for_completion(prompt_id: str, host: str) -> bool:
    """Poll /history until the prompt finishes. Returns False if timeout is hit."""
    elapsed = 0
    while elapsed < MAX_WAIT_SECONDS:
        try:
            resp = requests.get(f"http://{host}/history/{prompt_id}", timeout=5)
            history = resp.json()
            if prompt_id in history and history[prompt_id].get("outputs"):
                return True
        except requests.RequestException:
            pass
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    print(f"\n[WARN] Job {prompt_id} timed out after {MAX_WAIT_SECONDS}s — skipping.")
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch generate images via ComfyUI REST API")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode", choices=["surgeon", "clinic", "video", "all"],
        help="Use the structured JSON prompt library (surgeon / clinic / video / all)"
    )
    parser.add_argument("--workflow", type=Path, help="Override workflow JSON path")
    parser.add_argument("--prompts", type=Path, help="Plain-text prompts file (one per line)")
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducibility")
    parser.add_argument("--id", dest="prompt_id", help="Run only the prompt with this ID")
    args = parser.parse_args()

    # ── Resolve prompt list and workflow ─────────────────────────────────────
    jobs: list[tuple[dict, Path]] = []  # (prompt_dict, workflow_path)

    if args.mode:
        if args.mode in ("surgeon", "all"):
            if not SURGEON_PROMPTS.exists():
                print(f"[ERROR] Prompt library not found: {SURGEON_PROMPTS}")
                sys.exit(1)
            surgeon_prompts = load_json_prompts(SURGEON_PROMPTS)
            workflow = args.workflow or SURGEON_WORKFLOW
            jobs += [(p, workflow) for p in surgeon_prompts]

        if args.mode in ("clinic", "all"):
            if not CLINIC_PROMPTS.exists():
                print(f"[ERROR] Prompt library not found: {CLINIC_PROMPTS}")
                sys.exit(1)
            clinic_prompts = load_json_prompts(CLINIC_PROMPTS)
            workflow = args.workflow or CLINIC_WORKFLOW
            jobs += [(p, workflow) for p in clinic_prompts]

        if args.mode in ("video", "all"):
            if not VIDEO_PROMPTS.exists():
                print(f"[ERROR] Prompt library not found: {VIDEO_PROMPTS}")
                sys.exit(1)
            video_prompts = load_json_prompts(VIDEO_PROMPTS)
            workflow = args.workflow or VIDEO_WORKFLOW
            jobs += [(p, workflow) for p in video_prompts]

    elif args.prompts:
        if not args.prompts.exists():
            print(f"[ERROR] Prompts file not found: {args.prompts}")
            sys.exit(1)
        workflow = args.workflow or SURGEON_WORKFLOW
        jobs = [(p, workflow) for p in load_text_prompts(args.prompts)]

    else:
        parser.print_help()
        print("\n[ERROR] Specify either --mode or --prompts.")
        sys.exit(1)

    # ── Filter by --id if specified ──────────────────────────────────────────
    if args.prompt_id:
        jobs = [(p, wf) for p, wf in jobs if p.get("id") == args.prompt_id]
        if not jobs:
            print(f"[ERROR] No prompt found with id='{args.prompt_id}'")
            sys.exit(1)

    # ── Expand by --count ────────────────────────────────────────────────────
    jobs = [(p, wf) for p, wf in jobs for _ in range(args.count)]

    # ── Verify ComfyUI is reachable ──────────────────────────────────────────
    try:
        requests.get(f"http://{args.host}/system_stats", timeout=3)
    except requests.RequestException:
        print(f"[ERROR] Cannot reach ComfyUI at http://{args.host}")
        print("        Make sure ComfyUI is running (Run_Project.bat) before using this script.")
        sys.exit(1)

    # ── Run jobs ─────────────────────────────────────────────────────────────
    print(f"\nQueuing {len(jobs)} generation job(s) to ComfyUI at http://{args.host}\n")
    loaded_workflows: dict[Path, dict] = {}

    for prompt, wf_path in tqdm(jobs, unit="job"):
        if not wf_path.exists():
            print(f"\n[ERROR] Workflow not found: {wf_path} — skipping.")
            continue

        if wf_path not in loaded_workflows:
            loaded_workflows[wf_path] = load_workflow(wf_path)

        base_wf = loaded_workflows[wf_path]
        wf = inject_prompt(base_wf, prompt, seed=args.seed)

        label = prompt.get("label", prompt.get("id", "?"))
        try:
            prompt_id = queue_prompt(wf, args.host)
            tqdm.write(f"  [{prompt.get('id', '?')}] {label} → queued ({prompt_id[:8]}…)")
            wait_for_completion(prompt_id, args.host)
        except requests.RequestException as e:
            tqdm.write(f"\n[ERROR] Job failed ({label}): {e}")

    print("\nDone. Check ComfyUI's output folder for your generated images.")


if __name__ == "__main__":
    main()
