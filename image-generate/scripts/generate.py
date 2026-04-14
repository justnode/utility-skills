#!/usr/bin/env python3
"""
Image Generation Script using Replicate API.

Generates images from text prompts using supported Replicate models:
- google/nano-banana-2 (Gemini 3.1 Flash Image)
- bytedance/seedream-4.5 (cinematic aesthetics)
- wan-video/wan-2.7-image-pro (text-to-image generation)

Outputs one or more images per run, depending on model support.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# replicate is imported after argument parsing to fail fast on bad args

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "google/nano-banana-2": {
        "aspect_ratios": [
            "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9",
            "1:4", "4:1", "1:8", "8:1",
        ],
        "output_formats": ["jpg", "png"],
        "default_aspect_ratio": "1:1",
        "supports_output_format": True,
        "resolution_choices": ["512px", "1K", "2K", "4K"],
        "default_resolution": "1K",
    },
    "bytedance/seedream-4.5": {
        "aspect_ratios": ["1:1"],
        "default_aspect_ratio": "1:1",
        "supports_output_format": False,
        "size_choices": ["2K", "4K"],
        "default_size": "2K",
    },
    "wan-video/wan-2.7-image-pro": {
        "aspect_ratios": ["1:1"],
        "default_aspect_ratio": "1:1",
        "supports_output_format": False,
        "size_choices": ["1K", "2K", "4K"],
        "default_size": "2K",
        "supports_seed": True,
    },
}

DEFAULT_MODEL = "wan-video/wan-2.7-image-pro"
DEFAULT_FORMAT = "jpg"
GLOBAL_ENV_PATH = Path.home() / ".utility-skills" / ".env"
DEFAULT_MODEL_ENV_VAR = "IMAGE_GENERATE_DEFAULT_MODEL"

# API retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables win over values from the file so callers can
    still override configuration per shell/session.
    """
    if not env_path.exists():
        return

    try:
        env_lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Warning: failed to read env file {env_path}: {exc}", file=sys.stderr)
        return

    for raw_line in env_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def extract_output_files(output: object) -> list[object]:
    """Normalize Replicate output into a list of output files/URLs."""
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned an empty output list")
        return output

    return [output]


def detect_output_extension(file_output: object, fallback: str) -> str:
    """Infer an output extension from a Replicate file URL when possible."""
    output_url = ""

    if hasattr(file_output, "url"):
        output_url = getattr(file_output, "url")
    elif isinstance(file_output, str):
        output_url = file_output

    if output_url:
        suffix = Path(urlparse(output_url).path).suffix.lower().lstrip(".")
        if suffix in {"jpeg", "jpg", "png", "webp"}:
            return "jpg" if suffix == "jpeg" else suffix

    return fallback


def read_output_bytes(file_output: object) -> bytes:
    """Read raw image bytes from a Replicate output object."""
    if hasattr(file_output, "read"):
        return file_output.read()

    if isinstance(file_output, str):
        import urllib.request

        with urllib.request.urlopen(file_output) as resp:
            return resp.read()

    raise TypeError(f"Unexpected output type: {type(file_output)}")


def generate_images(
    model: str, input_params: dict, fallback_extension: str
) -> list[tuple[bytes, str]]:
    """Call Replicate API and return all generated images plus extensions.

    Args:
        model: Replicate model identifier.
        input_params: Model-specific Replicate input payload.
        fallback_extension: Extension to use when the output URL has no suffix.

    Returns:
        A list of `(raw_image_bytes, file_extension)` tuples.
    """
    import replicate

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Calling Replicate API (attempt {attempt}/{MAX_RETRIES}) …")
            output = replicate.run(model, input=input_params)
            file_outputs = extract_output_files(output)
            results: list[tuple[bytes, str]] = []

            for file_output in file_outputs:
                image_bytes = read_output_bytes(file_output)
                if not image_bytes:
                    raise RuntimeError("Received empty image data from API")
                results.append(
                    (image_bytes, detect_output_extension(file_output, fallback_extension))
                )

            return results

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


def resolve_default_model() -> str:
    """Resolve the default model from environment configuration."""
    configured_model = os.environ.get(DEFAULT_MODEL_ENV_VAR, "").strip()
    if not configured_model:
        return DEFAULT_MODEL

    if configured_model not in SUPPORTED_MODELS:
        print(
            f"Warning: {DEFAULT_MODEL_ENV_VAR}={configured_model!r} is not supported. "
            f"Falling back to {DEFAULT_MODEL}.",
            file=sys.stderr,
        )
        return DEFAULT_MODEL

    return configured_model


def build_parser(default_model: str) -> argparse.ArgumentParser:
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
        default=default_model,
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"Replicate model to use (default: {default_model})",
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
        default=None,
        help="Output image format for models that support it (google/nano-banana-2 only)",
    )
    parser.add_argument(
        "--resolution",
        choices=SUPPORTED_MODELS["google/nano-banana-2"]["resolution_choices"],
        default=None,
        help="Output resolution for google/nano-banana-2 (default: 1K)",
    )
    parser.add_argument(
        "--size",
        default=None,
        help=(
            "Model-specific output size. Seedream supports 2K/4K. "
            "Wan supports 1K/2K/4K or exact dimensions like '1920*1080'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for models that support it (wan-video/wan-2.7-image-pro only)",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help=(
            "Number of output images to generate when the selected model supports it. "
            "Seedream supports up to 15. Wan supports up to 4 in this skill."
        ),
    )
    return parser


def is_valid_wan_size(size: str) -> bool:
    """Return True when the Wan size value matches the documented schema."""
    if size in SUPPORTED_MODELS["wan-video/wan-2.7-image-pro"]["size_choices"]:
        return True

    return re.fullmatch(r"\d+\*\d+", size) is not None


def build_model_input(model: str, args: argparse.Namespace) -> tuple[dict, str, str]:
    """Build a model-specific Replicate input payload.

    Returns:
        Tuple of input params, resolved aspect ratio, and preferred file extension.
    """
    model_config = SUPPORTED_MODELS[model]

    if args.num_outputs < 1:
        raise ValueError("--num-outputs must be at least 1")

    if model == "google/nano-banana-2":
        aspect_ratio = args.aspect_ratio or model_config["default_aspect_ratio"]
        if aspect_ratio not in model_config["aspect_ratios"]:
            supported = ", ".join(model_config["aspect_ratios"])
            raise ValueError(
                f"Aspect ratio '{aspect_ratio}' is not supported by {model}. "
                f"Supported ratios: {supported}"
            )

        input_params: dict = {
            "prompt": args.prompt,
            "aspect_ratio": aspect_ratio,
        }

        output_format = args.output_format or DEFAULT_FORMAT
        input_params["output_format"] = output_format
        input_params["resolution"] = args.resolution or model_config["default_resolution"]

        if args.size:
            raise ValueError(
                "--size is only supported for bytedance/seedream-4.5 "
                "and wan-video/wan-2.7-image-pro"
            )
        if args.seed is not None:
            raise ValueError("--seed is not supported for google/nano-banana-2")
        if args.num_outputs != 1:
            raise ValueError("--num-outputs is not supported for google/nano-banana-2")

        return input_params, aspect_ratio, output_format

    if model == "bytedance/seedream-4.5":
        aspect_ratio = args.aspect_ratio or model_config["default_aspect_ratio"]
        if aspect_ratio not in model_config["aspect_ratios"]:
            supported = ", ".join(model_config["aspect_ratios"])
            raise ValueError(
                f"Aspect ratio '{aspect_ratio}' is not supported by {model}. "
                f"Supported ratios: {supported}"
            )

        input_params: dict = {
            "prompt": args.prompt,
            "aspect_ratio": aspect_ratio,
        }

        if args.output_format:
            raise ValueError(
                "--output-format is not supported for bytedance/seedream-4.5 "
                "according to the current Replicate API."
            )

        if args.resolution:
            raise ValueError("--resolution is only supported for google/nano-banana-2")

        if args.seed is not None:
            raise ValueError("--seed is not supported for bytedance/seedream-4.5")

        resolved_size = args.size or model_config["default_size"]
        if resolved_size not in model_config["size_choices"]:
            supported = ", ".join(model_config["size_choices"])
            raise ValueError(
                f"Size '{resolved_size}' is not supported by {model}. "
                f"Supported sizes: {supported}"
            )

        input_params["size"] = resolved_size
        if args.num_outputs > 15:
            raise ValueError(
                "--num-outputs exceeds the current seedream-4.5 limit of 15"
            )
        if args.num_outputs > 1:
            input_params["sequential_image_generation"] = "auto"
            input_params["max_images"] = args.num_outputs
        return input_params, aspect_ratio, DEFAULT_FORMAT

    if model == "wan-video/wan-2.7-image-pro":
        if args.output_format:
            raise ValueError("--output-format is not supported for wan-video/wan-2.7-image-pro")

        if args.resolution:
            raise ValueError("--resolution is only supported for google/nano-banana-2")

        if args.aspect_ratio:
            raise ValueError(
                "--aspect-ratio is not supported for wan-video/wan-2.7-image-pro. "
                "Use --size with presets like 1K/2K/4K or exact dimensions such as '1920*1080'."
            )

        input_params: dict = {
            "prompt": args.prompt,
        }
        resolved_size = args.size or model_config["default_size"]
        if not is_valid_wan_size(resolved_size):
            raise ValueError(
                f"Size '{resolved_size}' is not supported by {model}. "
                "Use 1K, 2K, 4K, or exact dimensions like '1920*1080'."
            )

        input_params["size"] = resolved_size
        if args.num_outputs > 4:
            raise ValueError(
                "--num-outputs exceeds the current wan-video/wan-2.7-image-pro limit "
                "supported by this skill (4)"
            )
        input_params["num_outputs"] = args.num_outputs

        if args.seed is not None:
            input_params["seed"] = args.seed

        return input_params, model_config["default_aspect_ratio"], DEFAULT_FORMAT

    raise ValueError(f"Unsupported model configuration for {model}")


def main() -> None:
    # Load shared user-level configuration before validating required variables.
    load_env_file(GLOBAL_ENV_PATH)
    default_model = resolve_default_model()
    parser = build_parser(default_model)
    args = parser.parse_args()

    # Validate API token
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print(
            "Error: REPLICATE_API_TOKEN environment variable is not set.\n"
            f"Please add it to {GLOBAL_ENV_PATH}, for example:\n"
            "REPLICATE_API_TOKEN=r8_...\n"
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

    try:
        input_params, aspect_ratio, preferred_extension = build_model_input(args.model, args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Display parameters
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          Image Generation Tool                  ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Model:        {args.model}")
    print(f"  Prompt:       {args.prompt}")
    print(f"  Aspect Ratio: {aspect_ratio}")
    print(f"  Outputs:      {args.num_outputs}")
    if args.model == "google/nano-banana-2":
        print(f"  Resolution:   {input_params['resolution']}")
        print(f"  Format:       {input_params['output_format']}")
    else:
        print(f"  Size:         {input_params['size']}")
    if "seed" in input_params:
        print(f"  Seed:         {input_params['seed']}")
    print(f"  Output Dir:   {output_dir}")
    print()

    # Generate images
    start_time = time.time()
    image_results = generate_images(args.model, input_params, preferred_extension)
    elapsed = time.time() - start_time
    model_short = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths: list[Path] = []
    total_size_kb = 0.0

    for index, (image_bytes, output_extension) in enumerate(image_results, start=1):
        if len(image_results) == 1:
            filename = f"{model_short}_{timestamp}.{output_extension}"
        else:
            filename = f"{model_short}_{timestamp}_{index:02d}.{output_extension}"

        output_path = output_dir / filename
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        output_paths.append(output_path)
        total_size_kb += len(image_bytes) / 1024

    print()
    print(f"  ✅ Generated {len(image_results)} image(s) successfully!")
    if len(output_paths) == 1:
        print(f"  📄 Saved to: {output_paths[0]}")
    else:
        for output_path in output_paths:
            print(f"  📄 Saved to: {output_path}")
    print(f"  📊 Total Size: {total_size_kb:.1f} KB")
    print(f"  ⏱  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
