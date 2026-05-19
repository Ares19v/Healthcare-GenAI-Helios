<div align="center">

# Healthcare-GenAI-Helios

**Custom LoRA Fine-Tuning · ComfyUI Automation · AI Healthcare Marketing Pipeline**

[![CI — Validate Pipeline](https://github.com/Ares19v/Healthcare-GenAI-Helios/actions/workflows/ci.yml/badge.svg)](https://github.com/Ares19v/Healthcare-GenAI-Helios/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Stable Diffusion 1.5](https://img.shields.io/badge/Base_Model-SD_1.5-7C3AED)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[![Kohya_ss](https://img.shields.io/badge/Training-Kohya__ss-F59E0B)](https://github.com/kohya-ss/sd-scripts)
[![WD14 Tagger](https://img.shields.io/badge/Auto--Tag-WD14_SwinV2-06B6D4)](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)


*A production-grade generative AI pipeline that fine-tunes Stable Diffusion on a specific doctor and clinic identity, then automates photorealistic healthcare marketing asset generation — brand-consistent, on demand, at zero marginal cost per image.*

</div>

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [What This Pipeline Does](#2-what-this-pipeline-does)
3. [Architecture](#3-architecture)
4. [Technical Deep-Dive](#4-technical-deep-dive)
5. [Dataset Engineering](#5-dataset-engineering)
6. [Training Configuration](#6-training-configuration)
7. [Quick Start](#7-quick-start)
8. [Image Generation](#8-image-generation)
9. [Prompt Library](#9-prompt-library)
10. [Automation Scripts](#10-automation-scripts)
11. [Project Structure](#11-project-structure)
12. [Retraining](#12-retraining)
13. [Roadmap](#13-roadmap)
14. [Acknowledgements](#14-acknowledgements)

---

## 1. The Problem

Healthcare marketing agencies face a costly and legally complex challenge when producing visual content for clinics and medical professionals:

- **Professional medical photography** costs ₹15,000–₹80,000 per shoot and requires scheduling a real doctor, a location, a photographer, and post-production
- **Stock photo libraries** produce generic, off-brand imagery that is impossible to differentiate between competing clinics
- **Generic AI generators** (Midjourney, DALL-E) produce a different face and a different clinic every single time — completely unusable for brand-consistent campaigns

This pipeline eliminates all three problems. By fine-tuning Stable Diffusion with LoRA on a specific doctor's synthetic portrait dataset and a specific clinic's interior/exterior photographs, the model **permanently memorizes both visual identities**. Two trigger words (`HeliosSurgeon`, `HeliosClinic`) reliably reproduce them in any lighting, pose, or composition — at zero cost per image.

---

## 2. What This Pipeline Does

```
Training Phase (one-time, ~1 hour on RTX 5060)
  24 doctor portraits   →  LoRA fine-tuning  →  Helios_OrthoJoint_v1.safetensors
  23 clinic interiors   →

Generation Phase (on demand, seconds per image)
  HeliosSurgeon + prompt  →  ComfyUI  →  photorealistic doctor portrait
  HeliosClinic  + prompt  →  ComfyUI  →  photorealistic clinic interior
  Both triggers together  →  ComfyUI  →  doctor in the clinic
```

**Key capabilities:**
- ✅ Consistent doctor portraits — any angle, expression, or lighting scenario
- ✅ Consistent clinic interiors — reception, operating theatre, corridors, consultation rooms
- ✅ Combined scenes — the same doctor placed inside the same clinic (both triggers simultaneously)
- ✅ **Video Generation** — AnimateDiff integration for dynamic short-form marketing clips
- ✅ Batch generation via Python API — structured prompt library with 19 curated presets
- ✅ Fully automated dataset captioning — one `Auto_Tag_WD14.bat` tags everything
- ✅ One-click setup — `install.bat` configures the full environment from scratch
- ✅ Pause/resume training — `save_state = true` allows stopping and continuing mid-run

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
│                                                                     │
│  Raw Images (47 total)                                              │
│       │                                                             │
│       ▼                                                             │
│  Auto_Tag_WD14.bat                                                  │
│  ├── WD14 SwinV2 ONNX tagger  → content tags per image              │
│  └── auto_caption.py          → prepends HeliosSurgeon/HeliosClinic │
│                                                                     │
│  dataset/20_HeliosSurgeon/  (24 portraits, trigger: HeliosSurgeon)  │
│  dataset/20_HeliosClinic/   (23 interiors, trigger: HeliosClinic)   │
│       │                                                             │
│       ▼                                                             │
│  Kohya_ss LoRA Training                                             │
│  ├── Base: SD 1.5 (v1-5-pruned-emaonly-fp16)                        │
│  ├── Optimizer: AdamW8bit | Precision: bf16 | Res: 512px            │
│  ├── Network: dim=32, alpha=16 | Steps: 2,820 | Epochs: 6           │
│  └── xformers memory-efficient attention enabled                    │
│       │                                                             │
│       ▼                                                             │
│  Helios_OrthoJoint_v1.safetensors  (~36 MB)                         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────┐
│                     GENERATION PHASE                                │
│                                                                     │
│  ComfyUI Studio                                                     │
│  ├── SD 1.5 base checkpoint (frozen, never modified)                │
│  ├── LoRA plugin: Helios_OrthoJoint_v1.safetensors                  │
│  ├── AnimateDiff (Optional): mm_sd_v15_v2.ckpt                      │
│  ├── KSampler: 30 steps · CFG 7 · dpmpp_2m · karras                 │
│  │                                                                  │
│  ├── Helios_Surgeon_v1.json  →  doctor portrait workflow            │
│  ├── Helios_Clinic_v1.json   →  clinic interior workflow            │
│  └── Helios_AnimateDiff_Txt2Vid_v1.json → video workflow            │
│                                                                     │
│  batch_generate.py  →  ComfyUI REST API  →  outputs/ folder         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Deep-Dive

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Rather than retraining Stable Diffusion's entire 860M-parameter UNet — which would produce a new 4 GB file and take days — LoRA injects two small trainable matrices (`A` and `B`) alongside the existing frozen weight matrices (`W₀`):

```
Output = W₀x + ΔWx,   where ΔW = B·A   (B ∈ ℝᵐˣʳ, A ∈ ℝʳˣⁿ, rank r ≪ n)
```

With `network_dim = 32` (rank = 32), the LoRA updates roughly **0.37% of the UNet's parameters**, producing a ~36 MB file that permanently alters the model's output for the trained concepts without touching the base weights. The base model stays completely clean and can be loaded by any other project.

### Why These Specific Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| `network_dim` (rank) | 32 | Enough capacity to independently memorize two distinct visual identities (face + clinic) without overfitting a small dataset. |
| `network_alpha` | 16 | Alpha controls the effective learning rate as `alpha/dim`. Setting `alpha = dim/2` is the industry standard — prevents the extreme weight updates that occur when alpha equals dim. |
| `resolution` | 512×512 | SD 1.5's native training resolution. Chosen for 8GB VRAM compatibility while still preserving facial and interior detail at this scale. |
| `mixed_precision` | bf16 | RTX 5060 (Ada Lovelace) has dedicated bf16 tensor cores. bf16 has a wider dynamic range than fp16 (8-bit exponent vs 5-bit), preventing gradient overflow in early high-loss training. |
| `optimizer` | AdamW8bit | Stores optimizer states in 8-bit integers — cuts VRAM usage by ~40% vs standard AdamW with no meaningful quality loss. Critical for fitting batch_size=2 on 8GB VRAM. |
| `lr_scheduler` | cosine_with_restarts | Cosine decay gradually reduces the learning rate, preventing the optimizer from making destructive large updates toward the end of training. |
| `epochs` | 6 | (24 surgeon + 23 clinic) × 20 repeats = 940 training images/epoch. batch_size=2 → 470 steps/epoch × 6 = 2,820 total steps. Sufficient for convergence on this dataset size. |

### Why WD14 Auto-Tagging (Not Just Trigger Words)

A naive approach would be to caption every image with just its trigger word (`HeliosSurgeon`). This causes **concept bleed** — the AI cannot distinguish which parts of the image the trigger word refers to. It would permanently associate `HeliosSurgeon` with the doctor's blue scrubs, the hospital background, and the stethoscope, making it impossible to later generate the doctor in different clothing or settings.

By running WD14 to exhaustively tag every other element in the image (`blue_shirt, stethoscope, indoors, white_wall`), the model can subtract all known concepts and assign only the unique, untagged remainder — the doctor's specific face — to the `HeliosSurgeon` trigger word. The result is a trigger word that means **only the face**, giving complete creative control over everything else.

---

## 5. Dataset Engineering

| Subset | Folder | Images | Repeats | Trigger Word | Caption Method |
|---|---|---|---|---|---|
| Surgeon | `20_HeliosSurgeon/` | 24 | 20× | `HeliosSurgeon` | WD14 SwinV2 ONNX auto-tag |
| Clinic | `20_HeliosClinic/` | 23 | 20× | `HeliosClinic` | WD14 SwinV2 ONNX auto-tag |
| **Total** | | **47** | | | **940 training images/epoch** |

### The Folder Naming Convention

The `20_` prefix tells Kohya_ss to repeat each image **20 times per epoch**, giving the model sufficient exposure to each photo without requiring a large raw image collection. This is the established standard for small-dataset LoRA training.

### Caption Structure

Each image has a paired `.txt` caption file generated and finalized by the `Auto_Tag_WD14.bat` pipeline. The trigger word is always the first token, followed by WD14-generated content tags.

**Example caption** (`surgeon_portrait_01.txt`):
```
HeliosSurgeon, solo, 1boy, male_focus, doctor, white_coat, stethoscope,
short_hair, looking_at_viewer, indoors, realistic, professional
```

**Example caption** (`clinic_reception_01.txt`):
```
HeliosClinic, indoors, no_humans, waiting_room, chairs, reception_desk,
white_walls, modern, clean, ambient_lighting, hospital
```

### Dataset Source

All training images are **AI-generated synthetic portraits and environments** (generated via Gemini Image Generation). Using synthetic data eliminates legal liability around real patient/doctor imagery and ensures full commercial freedom over the training assets.

---

## 6. Training Configuration

Full config: [`training/config_lora.toml`](training/config_lora.toml)

**Hardware:** HP Omen 16 · NVIDIA RTX 5060 8GB VRAM · Windows 11

```toml
# Network
network_module  = "networks.lora"
network_dim     = 32      # LoRA rank
network_alpha   = 16      # Effective LR scale = alpha/dim = 0.5

# Resolution & Bucketing
resolution       = "512,512"
enable_bucket    = true
min_bucket_reso  = 256
max_bucket_reso  = 768

# Optimizer
optimizer_type   = "AdamW8bit"
learning_rate    = 1e-4
unet_lr          = 1e-4
text_encoder_lr  = 5e-5
lr_scheduler     = "cosine_with_restarts"

# Training duration
max_train_epochs = 6
train_batch_size = 2

# Precision & Memory
mixed_precision  = "bf16"
xformers         = true
save_precision   = "bf16"
save_state       = true   # enables pause/resume
```

---

## 7. Quick Start

### Prerequisites

- [ComfyUI Desktop App](https://github.com/comfyanonymous/ComfyUI/releases) installed
- [Kohya_ss](https://github.com/kohya-ss/sd-scripts) installed at `C:\kohya_ss`
- `Helios_OrthoJoint_v1.safetensors` in `models/` (see [Retraining](#11-retraining) to train it yourself)
- Python 3.10+

### 1. Set Up the Environment (One-Time)

```bat
install.bat
```

This automatically creates a Python virtual environment and installs all script dependencies.

### 2. Tag Your Dataset

Drop your images into `dataset/20_HeliosSurgeon/` and/or `dataset/20_HeliosClinic/`, then run:

```bat
Auto_Tag_WD14.bat
```

This will:
1. Run WD14 SwinV2 (ONNX) to analyze every image and write its content tags into paired `.txt` files
2. Run `auto_caption.py` to prepend the correct trigger word (`HeliosSurgeon` or `HeliosClinic`) to every caption

### 3. Train the LoRA

```powershell
cd C:\kohya_ss
C:\kohya_ss\venv\Scripts\python.exe -m accelerate.commands.launch `
    --num_cpu_threads_per_process 2 sd-scripts\train_network.py `
    --config_file "C:\Users\<YourUsername>\Desktop\Projects\Healthcare-GenAI-Helios\training\config_lora.toml"
```

Training takes approximately **1 hour** on an RTX 5060. Six checkpoint files are saved automatically at the end of each epoch.

### 4. Resume Training (If Interrupted)

```powershell
C:\kohya_ss\venv\Scripts\python.exe -m accelerate.commands.launch `
    --num_cpu_threads_per_process 2 sd-scripts\train_network.py `
    --config_file "C:\...\training\config_lora.toml" `
    --resume "C:\...\models\Helios_OrthoJoint_v1-state-000003"
```

Replace `000003` with the epoch number of the last saved state folder in `models/`.

---

## 8. Asset Generation

### Manual (ComfyUI)

Copy `Helios_OrthoJoint_v1.safetensors` into ComfyUI's `models/loras/` folder, then load a workflow from the `workflows/` directory:

| Workflow | Use Case |
|---|---|
| `Helios_Surgeon_v1.json` | Generate doctor portraits |
| `Helios_Clinic_v1.json` | Generate clinic interiors |
| `Helios_AnimateDiff_Txt2Vid_v1.json` | Generate animated marketing videos |

To generate the **doctor inside the clinic**, use either workflow and combine both trigger words in the positive prompt:
```
HeliosSurgeon, HeliosClinic, doctor standing in reception area, professional, cinematic
```

### Batch (Python Script)

```bash
# All surgeon presets from the prompt library
python scripts/batch_generate.py --mode surgeon

# All clinic presets from the prompt library
python scripts/batch_generate.py --mode clinic

# All animated video presets
python scripts/batch_generate.py --mode video

# All presets (surgeon + clinic + video), 3 variations each, fixed seed
python scripts/batch_generate.py --mode all --count 3 --seed 42

# One specific preset by ID
python scripts/batch_generate.py --mode surgeon --id SURG_003

# Use a plain text prompts file with a specific workflow
python scripts/batch_generate.py \
    --prompts training/sample_prompts.txt \
    --workflow workflows/Helios_Clinic_v1.json
```

---

## 9. Prompt Library

All prompts live in `prompts/` and are consumed by `batch_generate.py`. Each entry has a unique `id`, `label`, `positive`, `negative`, `cfg`, `steps`, `sampler`, and `scheduler` — all injected directly into the ComfyUI workflow automatically.

| File | Presets | Coverage |
|---|---|---|
| `surgeon_prompts.json` | 8 | White coat, surgical scrubs, consultation desk, billboard portrait, golden hour, team photo |
| `clinic_prompts.json` | 8 | Reception, consultation room, waiting area, OR, corridor, exterior day/night, combined scene |
| `animatediff_prompts.json` | 3 | Dynamic animated shots (greeting, panning, exam) specifically tuned for AnimateDiff |

**Add your own preset** by adding an entry to the relevant JSON:

```json
{
  "id": "SURG_009",
  "label": "My Custom Shot",
  "positive": "HeliosSurgeon, doctor outdoors, city skyline background, golden hour, cinematic",
  "negative": "blurry, watermark, deformed",
  "cfg": 7.0,
  "steps": 30,
  "sampler": "dpmpp_2m",
  "scheduler": "karras"
}
```

The batch script picks it up automatically — no code changes needed.

---

## 10. Automation Scripts

### `Auto_Tag_WD14.bat` — Dataset Captioning Pipeline

The primary automation tool. Chains two processes together:

1. **WD14 SwinV2 ONNX Tagger** (via Kohya_ss `sd-scripts/finetune/tag_images_by_wd14_tagger.py`): Runs computer vision inference on every image in the dataset. Uses a neural network trained on millions of human-tagged images to assign confidence scores across ~10,000 semantic tags. Tags scoring above the 0.35 threshold are written to the paired `.txt` file.

2. **`auto_caption.py`**: Scans all `.txt` files in the dataset. Reads the folder name to determine which trigger word applies (e.g., `20_HeliosSurgeon` → `HeliosSurgeon`), then prepends the trigger word to the front of every caption file.

### `scripts/batch_generate.py` — Image Batch Generation

Full-featured batch generator with structured prompt library support:

```
--mode surgeon|clinic|all  → uses prompts/*.json library
--id SURG_003              → run one specific preset
--count 3                  → generate each prompt 3 times
--seed 42                  → fixed seed for reproducibility
--prompts file.txt         → use plain text prompts instead
```

Injects positive prompt, negative prompt, cfg, steps, sampler, and seed directly into the ComfyUI workflow graph before queuing. Includes timeout handling and graceful error recovery.

### `install.bat` — Environment Setup

Creates a Python venv at `.venv/` and installs all dependencies from `requirements.txt`. Only needs to be run once per machine.

> **Full training walkthrough:** See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)

---

## 11. Project Structure

```
Healthcare-GenAI-Helios/
│
├── .github/
│   └── workflows/ci.yml          → CI: Python lint, TOML/JSON validation,
│                                      prompt schema check, large-file guard
├── .gitignore                    → Excludes models, outputs, venv, logs
├── LICENSE                       → MIT
├── README.md                     → This file
├── install.bat                   → One-click Python environment setup
├── Auto_Tag_WD14.bat             → Full dataset captioning pipeline
├── Run_Project.bat               → Launches ComfyUI with LoRA pre-loaded
│
├── dataset/
│   ├── 20_HeliosSurgeon/         → 24 doctor portraits + .txt captions
│   └── 20_HeliosClinic/          → 23 clinic interiors + .txt captions
│
├── docs/
│   └── TRAINING_GUIDE.md         → Full training walkthrough & troubleshooting
│
├── models/                       → LoRA weights — gitignored, never committed
│   └── Helios_OrthoJoint_v1.safetensors
│
├── outputs/                      → Generated images — gitignored
│
├── prompts/
│   ├── surgeon_prompts.json      → 8 curated surgeon generation presets
│   └── clinic_prompts.json       → 8 curated clinic generation presets
│
├── scripts/
│   ├── auto_caption.py           → Prepends trigger words to caption files
│   └── batch_generate.py         → Batch image generation via ComfyUI REST API
│
├── training/
│   ├── config_lora.toml          → Full Kohya_ss training configuration
│   └── sample_prompts.txt        → Test prompts evaluated after each epoch
│
└── workflows/
    ├── Helios_Surgeon_v1.json    → ComfyUI workflow: doctor portrait
    ├── Helios_Clinic_v1.json     → ComfyUI workflow: clinic interior
    └── Helios_AnimateDiff_Txt2Vid_v1.json → ComfyUI workflow: video animation
```

---

## 11. Retraining

To expand the dataset and produce a stronger V2 model:

**1. Add new images**
```
dataset/20_HeliosSurgeon/new_portrait.png
dataset/20_HeliosClinic/new_room.png
```

**2. Re-run the tagging pipeline**
```bat
Auto_Tag_WD14.bat
```

**3. Re-run training** — same command as Quick Start Step 3. A fresh LoRA will be trained from scratch incorporating all images.

**4. Pick the best epoch**

Six checkpoint files are saved automatically (`Helios_OrthoJoint_v1-000001.safetensors` through `...-000006.safetensors`). Load each into ComfyUI and compare outputs. The best result is typically **epoch 4 or 5** — epoch 6 can occasionally show signs of overtraining.

---

## 12. Roadmap

- [x] **Prompt library** — structured JSON presets for common healthcare scenarios (consultation, surgery, branding shots)
- [x] **Video generation** — AnimateDiff integration for short animated clinic walkthroughs
- [ ] **ControlNet pose** — use OpenPose to precisely control the doctor's body position
- [ ] **Upscaling node** — chain Real-ESRGAN 4× into workflows for print-ready output
- [ ] **Extended dataset** — 50+ images covering diverse lighting, clothing, and clinic rooms

---

## 13. Acknowledgements

- **[Stability AI](https://stability.ai/)** — Stable Diffusion v1.5 base model
- **[kohya-ss](https://github.com/kohya-ss/sd-scripts)** — LoRA training framework
- **[comfyanonymous](https://github.com/comfyanonymous/ComfyUI)** — ComfyUI node-based generation interface
- **[SmilingWolf](https://huggingface.co/SmilingWolf)** — WD14 SwinV2 image tagger

---

<div align="center">

Built for the **Helios Healthcare** AI marketing pipeline · MIT License · 2026

</div>

---
<p align="center">
  Made by Devansh Tyagi @ 2026
</p>