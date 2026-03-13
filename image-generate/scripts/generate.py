#!/usr/bin/env python3
"""
Image Generation Script using Replicate API.

Generates images from text prompts using supported Replicate models:
- google/nano-banana-2 (Gemini 3.1 Flash Image)
- bytedance/seedream-4.5 (cinematic aesthetics)

Outputs one image per run in JPG (default) or PNG format.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# replicate is imported after argument parsing to fail fast on bad args

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "google/nano-banana-2": {
        "aspect_ratios": [
            "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9",
        ],
        "output_formats": ["jpg", "png"],
        "default_aspect_ratio": "1:1",
    },
    "bytedance/seedream-4.5": {
        "aspect_ratios": [
            "1:1", "3:2", "2:3", "16:9", "9:16",
            "4:3", "3:4", "21:9", "9:21",
        ],
        "output_formats": ["jpg", "png"],
        "default_aspect_ratio": "1:1",
    },
}

DEFAULT_MODEL = "google/nano-banana-2"
DEFAULT_FORMAT = "jpg"

# API retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def generate_image(
    model: str,
    prompt: str,
    aspect_ratio: str,
    output_format: str,
    seed: int | None = None,
) -> bytes:
    """Call Replicate API to generate an image and return raw bytes.

    Args:
        model: Replicate model identifier.
        prompt: Text prompt for image generation.
        aspect_ratio: Desired aspect ratio (e.g. "1:1", "16:9").
        output_format: Output format, "jpg" or "png".
        seed: Optional random seed for reproducibility.

    Returns:
        Raw image bytes.
    """
    import replicate

    input_params: dict = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
    }

    if seed is not None:
        input_params["seed"] = seed

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Calling Replicate API (attempt {attempt}/{MAX_RETRIES}) …")
            output = replicate.run(model, input=input_params)

            # Handle different output types from Replicate
            # Some models return a single FileOutput, others return a list
            if isinstance(output, list):
                file_output = output[0]
            else:
                file_output = output

            # FileOutput has a .read() method; plain string URLs need downloading
            if hasattr(file_output, "read"):
                image_bytes = file_output.read()
            elif isinstance(file_output, str):
                import urllib.request
                with urllib.request.urlopen(file_output) as resp:
                    image_bytes = resp.read()
            else:
                raise TypeError(f"Unexpected output type: {type(file_output)}")

            if not image_bytes:
                raise RuntimeError("Received empty image data from API")

            return image_bytes

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * attempt
                print(f"  ⚠ API error (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"    Retrying in {delay}s …")
                time.sleep(delay)
            else:
                break

    print(f"  ✗ API failed after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)
    raise last_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    model_names = ", ".join(SUPPORTED_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Generate images using Replicate API. "
                    f"Supported models: {model_names}.",
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Text prompt describing the image to generate",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"Replicate model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the generated image (default: current working directory)",
    )
    parser.add_argument(
        "-a", "--aspect-ratio",
        default=None,
        help="Aspect ratio, e.g. 1:1, 16:9, 9:16 (default: model's default, usually 1:1)",
    )
    parser.add_argument(
        "-f", "--output-format",
        choices=["jpg", "png"],
        default=DEFAULT_FORMAT,
        help=f"Output image format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate API token
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print(
            "Error: REPLICATE_API_TOKEN environment variable is not set.\n"
            "Please set it: export REPLICATE_API_TOKEN=r8_...\n"
            "Get a token from: https://replicate.com/account/api-tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate model
    model_config = SUPPORTED_MODELS.get(args.model)
    if not model_config:
        print(
            f"Error: Unsupported model '{args.model}'.\n"
            f"Supported models: {', '.join(SUPPORTED_MODELS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve aspect ratio
    aspect_ratio = args.aspect_ratio or model_config["default_aspect_ratio"]
    if aspect_ratio not in model_config["aspect_ratios"]:
        print(
            f"Error: Aspect ratio '{aspect_ratio}' is not supported by {args.model}.\n"
            f"Supported ratios: {', '.join(model_config['aspect_ratios'])}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename: {model_short}_{timestamp}.{format}
    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_short}_{timestamp}.{args.output_format}"
    output_path = output_dir / filename

    # Display parameters
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          Image Generation Tool                  ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Model:        {args.model}")
    print(f"  Prompt:       {args.prompt}")
    print(f"  Aspect Ratio: {aspect_ratio}")
    print(f"  Format:       {args.output_format}")
    if args.seed is not None:
        print(f"  Seed:         {args.seed}")
    print(f"  Output:       {output_path}")
    print()

    # Generate image
    start_time = time.time()
    image_bytes = generate_image(
        model=args.model,
        prompt=args.prompt,
        aspect_ratio=aspect_ratio,
        output_format=args.output_format,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    # Save image
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    size_kb = len(image_bytes) / 1024
    print()
    print(f"  ✅ Image generated successfully!")
    print(f"  📄 Saved to: {output_path}")
    print(f"  📊 Size: {size_kb:.1f} KB")
    print(f"  ⏱  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
