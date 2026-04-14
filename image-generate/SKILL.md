---
name: image-generate
description: >
  Generate images using Replicate API models. Supported models:
  google/nano-banana-2 (Gemini 3.1 Flash Image, fast generation with
  text rendering and multi-aspect-ratio support) and
  bytedance/seedream-4.5 (cinematic aesthetics with strong spatial
  understanding) and wan-video/wan-2.7-image-pro (4K-capable text-to-image
  generation). Outputs JPG by default, with multi-image support
  for selected models. Default model can be configured in ~/.utility-skills/.env.
  Requires REPLICATE_API_TOKEN in ~/.utility-skills/.env.
  Use when user asks to: "generate image", "create image", "text to image",
  "generate picture", "AI image", "生成图片", "AI绘图", "文生图",
  "replicate generate", "nano-banana", "seedream".
---

# Image Generate

Generate images via Replicate API. Supports `google/nano-banana-2`, `bytedance/seedream-4.5`, and `wan-video/wan-2.7-image-pro`.

## Setup

1. Determine this SKILL.md file's directory path as `SKILL_DIR`
2. Script path: `${SKILL_DIR}/scripts/generate.py`
3. Require `REPLICATE_API_TOKEN` in `~/.utility-skills/.env` (obtain from https://replicate.com/account/api-tokens). If not set, prompt the user to configure it before running.
4. Optional: read `IMAGE_GENERATE_DEFAULT_MODEL` from `~/.utility-skills/.env`. If it is not set, default to `wan-video/wan-2.7-image-pro`.

## Workflow

1. Determine the model to use:
   - **google/nano-banana-2** — fast, good text rendering, supports aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 1:4, 4:1, 1:8, 8:1; also supports `output_format` and `resolution`
   - **bytedance/seedream-4.5** — cinematic quality, strong spatial understanding; this skill currently uses the documented text-to-image subset with aspect ratio `1:1`, optional `size` (`2K` or `4K`), and multi-image output via `--num-outputs` up to 15
   - **wan-video/wan-2.7-image-pro** (default when not configured otherwise) — high-quality text-to-image generation; this skill currently exposes only the text-to-image subset with `size` (`1K`, `2K`, `4K`, or exact dimensions such as `1920*1080`), optional `seed`, and multi-image output via `--num-outputs` up to 4
2. Construct the prompt from user's description
3. Choose optional parameters that match the selected model's API
4. Run the script and deliver the saved image path to the user

## Usage

```bash
# Basic generation with the configured default model (wan by default)
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --prompt "A cat sitting on a windowsill"

# Specify model, aspect ratio, size, and output directory
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --model bytedance/seedream-4.5 --prompt "Cinematic sunset over mountains" --aspect-ratio 1:1 --size 2K --output-dir /path/to/save

# PNG format with higher resolution on nano-banana-2
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --prompt "A logo design" --output-format png --resolution 2K

# Wan with custom pixel dimensions
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --model wan-video/wan-2.7-image-pro --prompt "A cinematic sci-fi skyline at sunrise" --size '1920*1080' --seed 42

# Seedream multi-image generation
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --model bytedance/seedream-4.5 --prompt "A luxury watch campaign in a cinematic editorial style" --size 2K --num-outputs 4
```

Run `uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/generate.py --help` for all options.

## Key Details

- Output defaults to current working directory (where claude code was launched)
- File naming: `{model}_{timestamp}.{format}` for single-image runs, or `{model}_{timestamp}_{index}.{format}` for multi-image runs
- Only image generation is supported (no video)
- Some models can generate multiple images in one run; this skill saves every returned image
- Configuration is loaded from `~/.utility-skills/.env`
- `IMAGE_GENERATE_DEFAULT_MODEL` can override the default model selection
- Parameters are assembled per model so unsupported fields are not sent to Replicate
- Wan support in this skill is currently text-to-image only; editing inputs such as image sets are not exposed
- The script validates the model name and rejects unsupported models
- Aspect ratio support varies by model; invalid ratios will error with available options
- Wan uses `--size` instead of `--aspect-ratio`; quote values like `'1920*1080'` in the shell
