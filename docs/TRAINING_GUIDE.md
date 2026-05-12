# Healthcare-GenAI-Helios — Training Guide

A step-by-step guide for training the Helios LoRA model from scratch, understanding the results, and iterating to improve quality.

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Running automation scripts |
| Kohya_ss | latest | LoRA training engine |
| ComfyUI | latest | Image generation |
| NVIDIA GPU | 6GB+ VRAM | Training and inference |

---

## Step 1 — Prepare Your Dataset

Place your images in the correct folders:

```
dataset/
├── 20_HeliosSurgeon/     ← Doctor portrait images (.png / .jpg)
└── 20_HeliosClinic/      ← Clinic interior/exterior images
```

**Minimum recommended images:**
- Surgeon: 15–30 images. More angles = better flexibility. Avoid near-duplicate shots.
- Clinic: 15–30 images. Cover multiple rooms (reception, consultation, corridor, exterior).

**What makes a good dataset image:**
- ✅ Sharp, well-lit, in-focus subject
- ✅ Consistent identity (same doctor / same physical clinic)
- ✅ Variety of angles, poses, and lighting conditions
- ❌ Avoid blurry, heavily filtered, or low-resolution images
- ❌ Avoid watermarks or text overlays on the image

---

## Step 2 — Auto-Tag the Dataset

Run the tagging pipeline. This only needs to be run when you add new images:

```bat
Auto_Tag_WD14.bat
```

This script does two things in sequence:

1. **WD14 SwinV2 ONNX Tagger** — Analyses every image using a neural network and writes a `.txt` caption file next to each image. On the first run, it downloads the model (~300 MB) from Hugging Face automatically.

2. **auto_caption.py** — Opens every `.txt` file and prepends the correct trigger word (`HeliosSurgeon` or `HeliosClinic`) to the front.

**Verify the output:** Open a few `.txt` files in the dataset folders. Each should look like:

```
HeliosSurgeon, solo, 1boy, male_focus, white_coat, stethoscope, indoors, realistic
```

If the trigger word is at the front, everything is correct.

---

## Step 3 — Run Training

```powershell
cd C:\kohya_ss
C:\kohya_ss\venv\Scripts\python.exe -m accelerate.commands.launch `
    --num_cpu_threads_per_process 2 sd-scripts\train_network.py `
    --config_file "C:\Users\<YourUsername>\Desktop\Projects\Healthcare-GenAI-Helios\training\config_lora.toml"
```

**What you will see:**

```
epoch 1/6
steps:   8%|███ | 229/2820 [05:25<1:01:21, 1.42s/it, avr_loss=0.123]
```

- `avr_loss` should start around 0.13–0.15 and slowly decrease to around 0.10–0.12 by epoch 6.
- A loss that drops below 0.08 early is a sign of overfitting.
- Training takes approximately **1 hour** on an RTX 5060 (8GB VRAM).

**To pause training safely:** Press `Ctrl + C` in the terminal. Progress up to the last completed epoch is saved.

---

## Step 4 — Resume After Pause

```powershell
C:\kohya_ss\venv\Scripts\python.exe -m accelerate.commands.launch `
    --num_cpu_threads_per_process 2 sd-scripts\train_network.py `
    --config_file "C:\...\training\config_lora.toml" `
    --resume "C:\...\models\Helios_OrthoJoint_v1-state-000003"
```

Replace `000003` with the last epoch number visible in your `models/` folder.

---

## Step 5 — Pick the Best Epoch

After training completes, you will have 6 model files:

```
models/
├── Helios_OrthoJoint_v1-000001.safetensors
├── Helios_OrthoJoint_v1-000002.safetensors
├── Helios_OrthoJoint_v1-000003.safetensors
├── Helios_OrthoJoint_v1-000004.safetensors   ← usually best
├── Helios_OrthoJoint_v1-000005.safetensors   ← usually best
└── Helios_OrthoJoint_v1-000006.safetensors
```

Kohya also saves sample images at the end of each epoch (in `models/sample/`) using the prompts from `training/sample_prompts.txt`. Use these to visually compare the progression.

**What to look for:**
- The face should be consistent and recognisable across prompts
- The clinic should be identifiable even when you vary the lighting or composition
- Epoch 4 or 5 typically hits the sweet spot before overfitting begins at epoch 6

Copy the best epoch file to `models/Helios_OrthoJoint_v1.safetensors` (rename it) and then use that with ComfyUI.

---

## Step 6 — Use in ComfyUI

1. Copy your chosen `.safetensors` into ComfyUI's `models/loras/` folder.
2. Load a workflow from `workflows/`:
   - `Helios_Surgeon_v1.json` — for doctor portraits
   - `Helios_Clinic_v1.json` — for clinic interiors
3. In the workflow, your LoRA node should reference `Helios_OrthoJoint_v1`.
4. Hit **Queue Prompt**.

**LoRA strength:** The default strength is `1.0`. If the identity is too strong and looks artificial, lower it to `0.7–0.8`. If it is too weak and the face does not look like your doctor, raise it to `1.1–1.2`.

**AnimateDiff Note:** When using the `Helios_AnimateDiff_Txt2Vid_v1.json` workflow, you may need to lower the LoRA strength to `0.7–0.85`. Motion models (like `mm_sd_v15_v2.ckpt`) alter the latent space heavily, and a LoRA at `1.0` combined with motion can sometimes cause the video to look over-baked or visually "crispy."

---

## Understanding Loss

| Loss Range | Meaning |
|---|---|
| 0.14–0.16 | Normal starting loss — model is learning from noise |
| 0.11–0.13 | Healthy convergence — good generalisation |
| 0.09–0.11 | Strong learning — check sample images for quality |
| Below 0.08 | Potential overfit — the model is memorising, not learning |

Loss values in training are relative. The absolute number matters less than the **trend**. A smooth, gradual decrease is ideal.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARNING: No caption file found` | `caption_extension` mismatch | Ensure `config_lora.toml` has `caption_extension = ".txt"` |
| `CUDA out of memory` | Batch size too high for VRAM | Set `train_batch_size = 1` and `gradient_checkpointing = true` |
| Face is inconsistent | Too few training images | Add more images, especially varied angles |
| Face looks melted / distorted | Overtraining | Use an earlier epoch checkpoint |
| Trigger word generates wrong content | Concept bleed | Re-run `Auto_Tag_WD14.bat` and retrain |
| Training finishes in minutes | Only 1–2 images found | Check dataset folder path and image extensions |
