---
name: image-generate
description: >
  Generate images using Replicate API models. Supported models:
  google/nano-banana-2 (Gemini 3.1 Flash Image, fast generation with
  text rendering and multi-aspect-ratio support) and
  bytedance/seedream-4.5 (cinematic aesthetics with strong spatial
  understanding). Outputs JPG by default, one image per run.
  Requires REPLICATE_API_TOKEN environment variable.
  Use when user asks to: "generate image", "create image", "text to image",
  "generate picture", "AI image", "生成图片", "AI绘图", "文生图",
  "replicate generate", "nano-banana", "seedream".
---

# Image Generate

Generate images via Replicate API. Supports `google/nano-banana-2` and `bytedance/seedream-4.5`.

## Setup

1. Determine this SKILL.md file's directory path as `SKILL_DIR`
2. Script path: `${SKILL_DIR}/scripts/generate.py`
3. Require environment variable `REPLICATE_API_TOKEN` (obtain from https://replicate.com/account/api-tokens). If not set, prompt the user to set it before running.

## Workflow

1. Determine the model to use:
   - **google/nano-banana-2** (default) — fast, good text rendering, supports aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
   - **bytedance/seedream-4.5** — cinematic quality, strong spatial understanding, supports aspect ratios: 1:1, 3:2, 2:3, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21
2. Construct the prompt from user's description
3. Choose optional parameters: aspect ratio (default 1:1), output format (jpg/png, default jpg), seed
4. Run the script and deliver the saved image path to the user

## Usage

```bash
# Basic generation with default model (google/nano-banana-2)
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --prompt "A cat sitting on a windowsill"

# Specify model, aspect ratio, and output directory
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --model bytedance/seedream-4.5 --prompt "Cinematic sunset over mountains" --aspect-ratio 16:9 --output-dir /path/to/save

# PNG format with seed for reproducibility
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --prompt "A logo design" --output-format png --seed 42
```

Run `uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --help` for all options.

## Key Details

- Output defaults to current working directory (where claude code was launched)
- File naming: `{model}_{timestamp}.{format}` (e.g. `nano-banana-2_20260311_223000.jpg`)
- Only image generation is supported (no video)
- Each run generates exactly one image
- The script validates the model name and rejects unsupported models
- Aspect ratio support varies by model; invalid ratios will error with available options
