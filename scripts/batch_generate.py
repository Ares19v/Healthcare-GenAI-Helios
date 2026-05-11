"""
batch_generate.py
-----------------
Queues multiple image generation jobs to a running ComfyUI instance via
its local REST API. Reads prompts from a text file (one per line) and
sends each one as a separate queue entry using the specified workflow JSON.

Usage:
    .venv\\Scripts\\python.exe scripts\\batch_generate.py [options]

Options:
    --workflow   Path to a ComfyUI workflow JSON (default: workflows/Helios_Surgeon_v1.json)
    --prompts    Path to a prompts file, one prompt per line (default: training/sample_prompts.txt)
    --host       ComfyUI host (default: 127.0.0.1:8188)
    --count      Number of times to generate each prompt (default: 1)

Example:
    .venv\\Scripts\\python.exe scripts\\batch_generate.py ^
        --workflow workflows\\Helios_Clinic_v1.json ^
        --prompts training\\sample_prompts.txt ^
        --count 3
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
DEFAULT_WORKFLOW = Path(__file__).resolve().parent.parent / "workflows" / "Helios_Surgeon_v1.json"
DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "training" / "sample_prompts.txt"
POSITIVE_NODE_TITLE = "Positive Prompt"   # Must match the node title in your workflow JSON
POLL_INTERVAL = 3   # seconds between queue status checks


def load_workflow(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def find_positive_node_id(workflow: dict) -> str | None:
    """Find the node ID for the positive CLIPTextEncode node by its title."""
    for node_id, node in workflow.items():
        if isinstance(node, dict):
            meta = node.get("_meta", {})
            if meta.get("title", "").lower() == POSITIVE_NODE_TITLE.lower():
                return node_id
    return None


def queue_prompt(workflow: dict, host: str) -> str:
    """POST the workflow to ComfyUI and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"http://{host}/prompt", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["prompt_id"]


def wait_for_completion(prompt_id: str, host: str):
    """Poll /history until the prompt has finished executing."""
    while True:
        try:
            resp = requests.get(f"http://{host}/history/{prompt_id}", timeout=5)
            history = resp.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    return
        except requests.RequestException:
            pass
        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Batch generate images via ComfyUI API")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    # Validate files exist
    if not args.workflow.exists():
        print(f"[ERROR] Workflow not found: {args.workflow}")
        sys.exit(1)
    if not args.prompts.exists():
        print(f"[ERROR] Prompts file not found: {args.prompts}")
        sys.exit(1)

    # Verify ComfyUI is reachable
    try:
        requests.get(f"http://{args.host}/system_stats", timeout=3)
    except requests.RequestException:
        print(f"[ERROR] Cannot reach ComfyUI at http://{args.host}")
        print("        Make sure ComfyUI is running (Run_Project.bat) before using this script.")
        sys.exit(1)

    base_workflow = load_workflow(args.workflow)
    prompts = load_prompts(args.prompts)
    positive_node_id = find_positive_node_id(base_workflow)

    if not positive_node_id:
        print(f"[WARN] Could not find a node titled '{POSITIVE_NODE_TITLE}' in the workflow.")
        print("       Prompts will be queued without modifying the positive prompt node.")

    jobs = [(prompt, run_idx) for prompt in prompts for run_idx in range(args.count)]
    print(f"\nQueuing {len(jobs)} generation job(s) to ComfyUI at http://{args.host}\n")

    for prompt_text, run_idx in tqdm(jobs, unit="job"):
        workflow = json.loads(json.dumps(base_workflow))  # deep copy

        if positive_node_id:
            workflow[positive_node_id]["inputs"]["text"] = prompt_text

        # Randomise seed for variety
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = random.randint(0, 2**32 - 1)

        try:
            prompt_id = queue_prompt(workflow, args.host)
            wait_for_completion(prompt_id, args.host)
        except requests.RequestException as e:
            print(f"\n[ERROR] Job failed: {e}")

    print("Done. Check ComfyUI's output folder for your generated images.")


if __name__ == "__main__":
    main()
