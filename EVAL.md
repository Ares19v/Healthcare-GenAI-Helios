# EVAL — Healthcare-GenAI-Helios

> **Evaluation Date:** 2026-05-29  
> **Evaluator:** Automated Portfolio Review  
> **Maturity Level:** MVP

---

## 1. Project Purpose & Problem Statement

Healthcare marketing agencies face a specific and expensive problem: producing brand-consistent visual assets featuring a specific doctor or clinic. Professional photography costs ₹15,000–₹80,000 per shoot; stock libraries are generic; and general-purpose AI generators (Midjourney, DALL-E) produce a different face or environment every generation, making them useless for brand campaigns. 

Healthcare-GenAI-Helios solves this by fine-tuning Stable Diffusion 1.5 with LoRA on 47 synthetically-generated training images (24 doctor portraits, 23 clinic interiors), permanently encoding two trigger words — `HeliosSurgeon` and `HeliosClinic` — that reliably reproduce the same doctor and clinic identity in any pose, lighting, or setting. The pipeline also includes AnimateDiff integration for short-form video generation, completing a full image + video marketing asset workflow.

The use of AI-generated synthetic training data (via Gemini Image Generation) is a deliberate legal choice that eliminates liability around real medical professional imagery.

---

## 2. Technical Architecture

The pipeline has two distinct phases:

**Training Phase:**
1. 47 synthetic images deposited into two Kohya_ss dataset folders (`20_HeliosSurgeon/`, `20_HeliosClinic/`)
2. `Auto_Tag_WD14.bat` runs WD14 SwinV2 ONNX tagger to generate content tags for each image, then `auto_caption.py` prepends the correct trigger word to every `.txt` caption file — preventing concept bleed
3. Kohya_ss `sd-scripts/train_network.py` fine-tunes a LoRA network (rank=32, alpha=16) against the SD 1.5 base at 512×512 resolution for 6 epochs / 2,820 total steps
4. Output: `Helios_OrthoJoint_v1.safetensors` (~36 MB)

**Generation Phase:**
1. ComfyUI loads SD 1.5 + LoRA plugin
2. `batch_generate.py` calls the ComfyUI REST API, injects structured prompts from `surgeon_prompts.json` / `clinic_prompts.json` / `animatediff_prompts.json` by node class type (not hardcoded IDs — a robust design)
3. AnimateDiff (`mm_sd_v15_v2.ckpt`) enables text-to-video generation for marketing clips
4. Outputs saved to `outputs/` folder

---

## 3. Model / Algorithm Details

**Base Model:** Stable Diffusion v1.5 (2.13 GB, frozen — never modified)

**Fine-tuning Method:** LoRA (Low-Rank Adaptation) via Kohya_ss sd-scripts

**Key Hyperparameters:**
| Parameter | Value | Reasoning |
|---|---|---|
| LoRA rank (`network_dim`) | 32 | Sufficient capacity for two independent visual identities |
| LoRA alpha | 16 | alpha/dim = 0.5, industry standard for stable training |
| Resolution | 512×512 | SD 1.5 native resolution; 8GB VRAM compatible |
| Optimizer | AdamW8bit | ~40% VRAM reduction vs standard AdamW |
| Precision | bf16 | Wider dynamic range than fp16; native to RTX 5060 |
| LR scheduler | cosine_with_restarts | Prevents destructive updates in late training |
| Epochs | 6 | (47 images × 20 repeats) = 940/epoch; 6 epochs = 2,820 steps |
| Total parameters trained | ~0.37% of UNet | LoRA injects minimal adapters alongside frozen weights |

**Dataset Engineering:** The WD14 auto-tagging strategy is technically sound. Tagging all non-face attributes (stethoscope, white coat, background) allows the model to assign `HeliosSurgeon` exclusively to facial identity, giving full compositional control over clothing and setting. Without this, trigger words overfit to incidental attributes.

**Output model size:** ~36 MB (vs 4 GB full fine-tune) — parameter-efficient by design.

---

## 4. Strengths

- **WD14 auto-tagging pipeline** — preventing concept bleed via exhaustive attribute captioning is the correct methodology for identity LoRA training; this is well-understood and properly implemented.
- **Dual trigger word design** — separate `HeliosSurgeon` and `HeliosClinic` concepts that can be combined in a single prompt for "doctor inside clinic" scenes is a powerful compositional capability.
- **Synthetic training data** — eliminates legal and consent issues with real medical professional imagery; a thoughtful professional decision.
- **Structured prompt library** — 19 curated JSON presets with parameterized cfg/steps/sampler are immediately usable by a non-technical marketing team.
- **AnimateDiff integration** — extending stills to short animated clips is a genuine value-add for social media marketing.
- **Pause/resume training** — `save_state = true` in the Kohya config is an important operational feature for long training runs.
- **Detailed technical README** — hyperparameter table with explicit reasoning is above average; shows genuine understanding of why each choice was made.
- **GitHub Actions CI** — Python lint, TOML/JSON validation, prompt schema check, large-file guard.

---

## 5. Limitations & Known Gaps

- **No quantitative quality metrics.** There are no FID scores, CLIP-image similarity measurements, or identity preservation metrics (e.g., face embedding cosine similarity across generated variants). The evaluation is entirely qualitative.
- **Small dataset (47 images).** While the 20× repeat strategy partially compensates, 47 unique training images is a minimal dataset. Diversity of doctor poses, lighting conditions, and clinic rooms is limited.
- **SD 1.5 base model is aging.** The industry has substantially moved to SDXL, Flux, and SDXL-Lightning. SD 1.5 outputs look visibly softer and lower-resolution compared to modern alternatives, limiting the commercial viability of the output assets.
- **No ControlNet integration.** Precise pose control (from OpenPose) is on the roadmap but not implemented — without it, generating specific doctor postures for advertising layouts is difficult.
- **Resolution ceiling at 512×512.** Marketing assets for print or large-format digital need significantly higher resolution; Real-ESRGAN upscaling is planned but absent.
- **ComfyUI dependency is a heavy prerequisite.** The pipeline requires a separate ComfyUI installation, which adds setup friction for new users.
- **No output quality filtering.** Generated images are saved to `outputs/` without any automated quality gate — hallucinated faces or broken anatomy require manual review.
- **Model weights gitignored** — correct, but there is no automated download script; users must train from scratch.

---

## 6. Code Quality Assessment

**Structure:** Well-organized with clear separation between `dataset/`, `docs/`, `models/`, `prompts/`, `scripts/`, `training/`, and `workflows/`. The `batch_generate.py` design — injecting prompts by node class type rather than hardcoded IDs — is robust to ComfyUI workflow changes.

**Documentation:** The README is one of the most technically detailed in the portfolio, with explicit hyperparameter justifications, dataset folder math, and a caption engineering explanation. `TRAINING_GUIDE.md` in `docs/` provides the operational walkthrough.

**Test Coverage:** CI validates schema and file structure; no functional inference tests.

**Security:** `.gitignore` excludes models, outputs, and `PREP.txt`. No API keys in the generation pipeline (ComfyUI runs locally).

---

## 7. Maturity Breakdown

| Dimension | Score | Notes |
|-----------|-------|-------|
| Functionality | 7/10 | Training + generation pipeline functional; video works with AnimateDiff |
| Code Quality | 7/10 | Clean structure; batch scripts are robust |
| Documentation | 9/10 | Exceptional technical depth; best-documented training pipeline in portfolio |
| Scalability | 5/10 | SD 1.5 quality ceiling; no automated quality filtering; single-machine only |
| Security | 8/10 | Properly excludes sensitive files; synthetic training data avoids legal issues |
| **Overall** | **7.2/10** | Strong concept and execution; limited by SD 1.5 and absence of quality metrics |

---

## 8. Suggested Next Steps

1. **Migrate to SDXL or Flux base model.** SD 1.5 is three generations behind the current state of the art. Retraining on SDXL would dramatically improve output photorealism and detail — critical for professional healthcare marketing assets.
2. **Add quantitative identity preservation metrics.** Use a face embedding model (e.g., ArcFace or InsightFace) to compute cosine similarity between training portraits and generated outputs. This transforms "it looks right" into a measurable benchmark.
3. **Implement ControlNet pose guidance.** Even basic OpenPose integration would allow generating specific body positions for advertising layouts — a key missing capability for production use.

---

## 9. Verdict

Healthcare-GenAI-Helios is a genuinely practical application of generative AI to a real commercial problem — brand-consistent medical professional imagery — with a technically sound approach to the core challenge (WD14 auto-tagging for concept isolation, dual trigger word design). The documentation quality is exceptional, with unusually honest hyperparameter justifications. The main limitations are the aging SD 1.5 base model, the small training dataset, and the absence of quantitative quality metrics. A migration to a modern backbone and some face-identity preservation measurement would significantly strengthen this project's commercial and portfolio credibility.
