# Image Generate

Generate one or more images with Replicate models from this skill.

Supported models:

- `google/nano-banana-2`
- `bytedance/seedream-4.5`
- `wan-video/wan-2.7-image-pro`

Model-specific options:

- `google/nano-banana-2`: single-image only; supports `--aspect-ratio`, `--output-format`, and `--resolution`
- `bytedance/seedream-4.5`: supports `--aspect-ratio` (`1:1` only in this skill), `--size`, and `--num-outputs` up to 15
- `wan-video/wan-2.7-image-pro`: text-to-image only in this skill; supports `--size`, optional `--seed`, and `--num-outputs` up to 4

## Environment Variable Setup

This skill reads `REPLICATE_API_TOKEN` from the global user config file:

`~/.utility-skills/.env`

Create the directory and file:

```bash
mkdir -p ~/.utility-skills
touch ~/.utility-skills/.env
```

Add your Replicate token:

```dotenv
REPLICATE_API_TOKEN=r8_your_token_here
```

Optional: set the default model used when you do not pass `--model`:

```dotenv
IMAGE_GENERATE_DEFAULT_MODEL=wan-video/wan-2.7-image-pro
```

You can also use:

```dotenv
export REPLICATE_API_TOKEN=r8_your_token_here
```

Get a token from:

`https://replicate.com/account/api-tokens`

## Notes

- Shell environment variables still work and take precedence over the `.env` file.
- The script only needs `REPLICATE_API_TOKEN`; extra variables in `~/.utility-skills/.env` are ignored unless another skill uses them.
- `IMAGE_GENERATE_DEFAULT_MODEL` can be set to `google/nano-banana-2`, `bytedance/seedream-4.5`, or `wan-video/wan-2.7-image-pro`.
- Output files are saved to the current working directory by default, or to `--output-dir` if you provide one.

## Usage

From this skill directory:

```bash
uv run --project . python scripts/generate.py --prompt "A cat sitting on a windowsill"
```

With no `--model` flag, the script uses `IMAGE_GENERATE_DEFAULT_MODEL` from `~/.utility-skills/.env`. If that variable is not set, it defaults to `wan-video/wan-2.7-image-pro`.

Specify a model and aspect ratio:

```bash
uv run --project . python scripts/generate.py \
  --model bytedance/seedream-4.5 \
  --prompt "Cinematic sunset over mountains" \
  --aspect-ratio 1:1 \
  --size 2K
```

Use Nano Banana with PNG output:

```bash
uv run --project . python scripts/generate.py \
  --prompt "A logo design" \
  --output-format png \
  --resolution 2K \
  --output-dir /path/to/save
```

Use Wan with a fixed 16:9 pixel size:

```bash
uv run --project . python scripts/generate.py \
  --model wan-video/wan-2.7-image-pro \
  --prompt "A cinematic sci-fi skyline at sunrise" \
  --size '1920*1080' \
  --seed 42
```

Generate multiple Seedream images in one run:

```bash
uv run --project . python scripts/generate.py \
  --model bytedance/seedream-4.5 \
  --prompt "A luxury watch campaign in a cinematic editorial style" \
  --size 2K \
  --num-outputs 4
```

## Current parameter mapping

- `google/nano-banana-2` sends `prompt`, `aspect_ratio`, `output_format`, and `resolution`
- `bytedance/seedream-4.5` sends `prompt`, `aspect_ratio`, `size`, and for multi-image runs `sequential_image_generation=auto` plus `max_images`
- `wan-video/wan-2.7-image-pro` sends `prompt`, `size`, `num_outputs`, and optional `seed`
- `wan-video/wan-2.7-image-pro` uses `--size` instead of `--aspect-ratio`; for non-square output, pass exact dimensions like `'1920*1080'`
- `wan-video/wan-2.7-image-pro` editing-related inputs such as image sets are not exposed by this skill yet
- The script does not send `seed` for `google/nano-banana-2` or `bytedance/seedream-4.5` because their current Replicate APIs do not expose that field
- When a model returns multiple images, the script saves all of them. Filenames use `_01`, `_02`, etc.

Show all options:

```bash
uv run --project . python scripts/generate.py --help
```
