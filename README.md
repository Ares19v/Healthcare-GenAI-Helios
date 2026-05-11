<div align="center">

# 🏥 Healthcare-GenAI-Helios

**A custom AI production pipeline for brand-consistent, photorealistic medical marketing.**

[![CI](https://github.com/YOUR_GITHUB_USERNAME/Healthcare-GenAI-Helios/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/Healthcare-GenAI-Helios/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Generation_Engine-purple)](https://github.com/comfyanonymous/ComfyUI)
[![Kohya_ss](https://img.shields.io/badge/Kohya__ss-LoRA_Training-orange)](https://github.com/bmaltais/kohya_ss)

</div>

---

## Overview

Standard AI image generators are fundamentally unpredictable. Prompt the same description twice and you get two completely different faces and clinic layouts — incompatible with professional brand-consistent marketing.

**Healthcare-GenAI-Helios** solves this for **Helios OrthoJoint** — a specialist knee surgery practice — by training a custom LoRA (Low-Rank Adaptation) model that permanently memorises the visual identity of the surgeon and clinic. The result is an automated engine to generate infinite photorealistic marketing assets, with a consistent and recognisable brand aesthetic every single time.

---

## Architecture & Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LoRA Training** | Kohya_ss | Fine-tuning the base model on Helios imagery |
| **Base Model** | Stable Diffusion v1.5 | The foundational generalist AI artist |
| **Generation Engine** | ComfyUI | Node-based workflow interface for image production |
| **Hardware** | NVIDIA RTX 5060 (8GB VRAM) | Local training and inference |
| **Precision** | bf16 mixed precision | Native RTX 5060 hardware support |
| **Optimizer** | AdamW8bit | Memory-efficient weight adjustment |

---

## How It Works

```
Raw Photos → Caption & Tag → Kohya_ss Training → Helios LoRA → ComfyUI Pipeline → Marketing Assets
```

1. **Dataset:** Curated photos of the surgeon and clinic, captioned with trigger words
2. **Training:** The LoRA model learns *only* the visual identity of Helios from these photos
3. **Generation:** Trigger words activate the memorised identity at inference time
4. **Output:** Photorealistic, brand-consistent PNGs saved to `/outputs`

---

## Trigger Words

| Keyword | Activates |
|---|---|
| `HeliosSurgeon` | The surgeon's face, posture, and professional identity |
| `HeliosClinic` | The clinic's interior design, colour palette, and brand environment |

**Example prompts:**
```
HeliosSurgeon, professional surgeon portrait, blue surgical scrubs, warm studio lighting, 8k, photorealistic
HeliosClinic, modern orthopaedic clinic interior, reception desk, warm ambient lighting, 8k, architectural photography
```

---

## Training Parameters

| Parameter | Value | Reason |
|---|---|---|
| `network_dim` | 32 | Sufficient capacity for two distinct visual identities |
| `network_alpha` | 16 | Half of dim — industry-standard stability ratio |
| `resolution` | 768×768 | Preserves fine facial and architectural detail |
| `mixed_precision` | bf16 | Native to RTX 5060; more stable than fp16 |
| `optimizer` | AdamW8bit | VRAM-efficient without quality loss |
| `batch_size` | 2 | Optimal for 8GB VRAM |
| `max_train_epochs` | 6 | Best quality typically at epoch 4–5 |

---

## Directory Structure

```text
Healthcare-GenAI-Helios/
├── .github/
│   └── workflows/
│       └── ci.yml              ← GitHub Actions CI pipeline
├── dataset/
│   ├── 20_HeliosSurgeon/       ← Surgeon photos + .txt caption files
│   └── 20_HeliosClinic/        ← Clinic photos + .txt caption files
├── models/                     ← Trained LoRA .safetensors (git-ignored)
├── outputs/                    ← Generated marketing assets (git-ignored)
├── scripts/
│   ├── auto_caption.py         ← Auto-generates trigger-word caption stubs
│   └── batch_generate.py       ← Batch image generation via ComfyUI API
├── training/
│   ├── config_lora.toml        ← Kohya_ss training configuration
│   └── sample_prompts.txt      ← Prompts checked at each training epoch
├── workflows/
│   ├── Helios_Surgeon_v1.json  ← ComfyUI surgeon generation workflow
│   └── Helios_Clinic_v1.json   ← ComfyUI clinic generation workflow
├── .gitignore
├── install.bat                 ← Sets up Python environment
├── uninstall.bat               ← Tears down Python environment
├── LICENSE
├── README.md
├── requirements.txt
└── Run_Project.bat             ← One-click launch: syncs files + opens ComfyUI
```

---

## Quick Start

### 1. Install
```bat
install.bat
```

### 2. Prepare Dataset
Drop images into `dataset/20_HeliosSurgeon/` and `dataset/20_HeliosClinic/`, then:
```bat
.venv\Scripts\python.exe scripts\auto_caption.py
```
Then use Kohya_ss's built-in **WD14 Captioner** (Utilities tab) to enrich the captions with detail tags.

### 3. Train
```bat
cd C:\kohya_ss
C:\kohya_ss\venv\Scripts\python.exe -m accelerate.commands.launch ^
    --num_cpu_threads_per_process 2 sd-scripts\train_network.py ^
    --config_file "C:\Users\Devansh Tyagi\Desktop\Projects\Healthcare-GenAI-Helios\training\config_lora.toml"
```

### 4. Launch & Generate
```bat
Run_Project.bat
```
Copies the trained LoRA into ComfyUI, loads the workflows, and opens `http://127.0.0.1:8188`.

### 5. Batch Generate (Optional)
```bat
.venv\Scripts\python.exe scripts\batch_generate.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Output looks generic | Increase LoRA strength in LoraLoader node (try `1.2`) |
| Output looks distorted / burned | Lower LoRA strength (try `0.8`) |
| Wrong room / space generated | Verify caption files start with `HeliosClinic,` not `HeliosSurgeon,` |
| CUDA out of memory during training | Reduce `train_batch_size` to `1` in `training/config_lora.toml` |

---

## License

[MIT](LICENSE) © 2026 Devansh Tyagi
