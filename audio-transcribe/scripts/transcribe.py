#!/usr/bin/env python3
"""
Audio/Video Transcription Script using Groq Whisper API.

Transcribes video or audio files to timestamped text.
- Automatically extracts audio from video files using ffmpeg.
- Supports large file chunking (>25MB) with configurable chunk duration.
- Outputs in Markdown or TXT format with timestamps.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp",
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".ogg",
    ".aac", ".wma", ".opus", ".webm",
}

# Groq free-tier file size limit (bytes)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

DEFAULT_CHUNK_MINUTES = 10
DEFAULT_MODEL = "whisper-large-v3"
DEFAULT_FORMAT = "md"
DEFAULT_GRANULARITY = "segment"


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _get_ffmpeg_path() -> str:
    """Return the path to ffmpeg binary. Prefer global, fallback to static-ffmpeg."""
    global_ffmpeg = shutil.which("ffmpeg")
    if global_ffmpeg:
        return global_ffmpeg

    # Fallback: use static-ffmpeg package
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        path = shutil.which("ffmpeg")
        if path:
            return path
    except ImportError:
        pass

    print(
        "Error: ffmpeg not found. Install ffmpeg globally or run 'uv sync' to install static-ffmpeg.",
        file=sys.stderr,
    )
    sys.exit(1)


def _get_ffprobe_path() -> str:
    """Return the path to ffprobe binary."""
    global_ffprobe = shutil.which("ffprobe")
    if global_ffprobe:
        return global_ffprobe

    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        path = shutil.which("ffprobe")
        if path:
            return path
    except ImportError:
        pass

    print("Error: ffprobe not found.", file=sys.stderr)
    sys.exit(1)


def get_audio_duration(file_path: str, ffprobe_path: str) -> float:
    """Get the duration of an audio/video file in seconds."""
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_audio(
    input_path: str, output_path: str, ffmpeg_path: str,
    start_seconds: float | None = None, duration_seconds: float | None = None,
) -> None:
    """Extract audio from a video/audio file, downsampled to 16KHz mono FLAC."""
    cmd = [ffmpeg_path, "-y"]

    if start_seconds is not None:
        cmd += ["-ss", str(start_seconds)]
    if duration_seconds is not None:
        cmd += ["-t", str(duration_seconds)]

    cmd += [
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-map", "0:a",
        "-c:a", "flac",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Groq transcription
# ---------------------------------------------------------------------------


def transcribe_audio(
    client: Groq,
    audio_path: str,
    model: str,
    language: str | None,
    granularity: str,
) -> dict:
    """Transcribe a single audio file via Groq API and return the raw response."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format="verbose_json",
            timestamp_granularities=[granularity],
            language=language if language else None,  # None lets Whisper auto-detect
            temperature=0.0,
        )

    # The SDK returns a Pydantic model; convert to dict for uniform handling
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return json.loads(json.dumps(response, default=str))


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def process_file(
    input_path: str,
    client: Groq,
    model: str,
    language: str | None,
    granularity: str,
    chunk_minutes: int,
    ffmpeg_path: str,
    ffprobe_path: str,
) -> list[dict]:
    """
    Process an input file: extract audio if needed, chunk if large, transcribe.

    Returns a list of segment dicts with keys: start, end, text.
    """
    input_ext = Path(input_path).suffix.lower()
    is_video = input_ext in VIDEO_EXTENSIONS

    with tempfile.TemporaryDirectory(prefix="transcribe_") as tmp_dir:
        # Step 1: Extract / prepare audio as FLAC
        prepared_audio = os.path.join(tmp_dir, "audio.flac")

        if is_video:
            print(f"  Extracting audio from video …")
            extract_audio(input_path, prepared_audio, ffmpeg_path)
        else:
            # Even for audio files, re-encode to 16KHz mono FLAC for consistency
            print(f"  Preparing audio (16KHz mono FLAC) …")
            extract_audio(input_path, prepared_audio, ffmpeg_path)

        file_size = os.path.getsize(prepared_audio)

        # Step 2: Decide whether to chunk
        if file_size <= MAX_FILE_SIZE_BYTES:
            print(f"  File size: {file_size / 1024 / 1024:.1f} MB (within limit, no chunking needed)")
            print(f"  Transcribing …")
            result = transcribe_audio(client, prepared_audio, model, language, granularity)
            return _extract_segments(result, granularity, offset=0.0)
        else:
            print(f"  File size: {file_size / 1024 / 1024:.1f} MB (exceeds {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB limit)")
            duration = get_audio_duration(prepared_audio, ffprobe_path)
            chunk_duration = chunk_minutes * 60
            num_chunks = math.ceil(duration / chunk_duration)
            print(f"  Duration: {_fmt_seconds(duration)} | Splitting into {num_chunks} chunk(s) …")

            all_segments: list[dict] = []

            for i in range(num_chunks):
                start = i * chunk_duration
                chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.flac")

                # Determine actual chunk length (last chunk may be shorter)
                remaining = duration - start
                actual_duration = min(chunk_duration, remaining)

                print(f"  Chunk {i + 1}/{num_chunks}: {_fmt_seconds(start)} – {_fmt_seconds(start + actual_duration)}")
                extract_audio(prepared_audio, chunk_path, ffmpeg_path, start_seconds=start, duration_seconds=actual_duration)

                result = transcribe_audio(client, chunk_path, model, language, granularity)
                segments = _extract_segments(result, granularity, offset=start)
                all_segments.extend(segments)

            return all_segments


def _extract_segments(data: dict, granularity: str, offset: float) -> list[dict]:
    """Extract timestamped segments from the Groq API response."""
    segments: list[dict] = []

    if granularity == "word":
        words = data.get("words", [])
        if words:
            for w in words:
                segments.append({
                    "start": (w.get("start", 0) or 0) + offset,
                    "end": (w.get("end", 0) or 0) + offset,
                    "text": w.get("word", "").strip(),
                })
        else:
            # Fallback: treat entire text as one segment
            segments.append({
                "start": offset,
                "end": offset,
                "text": data.get("text", "").strip(),
            })
    else:
        # segment granularity
        segs = data.get("segments", [])
        if segs:
            for s in segs:
                segments.append({
                    "start": (s.get("start", 0) or 0) + offset,
                    "end": (s.get("end", 0) or 0) + offset,
                    "text": s.get("text", "").strip(),
                })
        else:
            segments.append({
                "start": offset,
                "end": offset,
                "text": data.get("text", "").strip(),
            })

    return segments


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_markdown(
    segments: list[dict],
    source_path: str,
    model: str,
    language: str | None,
) -> str:
    """Format transcription segments as a Markdown document."""
    filename = Path(source_path).name
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lang_display = language if language else "auto-detect"

    lines = [
        f"# Transcript: {filename}",
        "",
        f"- **Source**: {source_path}",
        f"- **Date**: {now}",
        f"- **Model**: {model}",
        f"- **Language**: {lang_display}",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]

    for seg in segments:
        ts = _fmt_seconds(seg["start"])
        text = seg["text"]
        if text:
            lines.append(f"[{ts}] {text}")
            lines.append("")

    return "\n".join(lines)


def format_txt(segments: list[dict]) -> str:
    """Format transcription segments as plain text with timestamps."""
    lines = []
    for seg in segments:
        ts = _fmt_seconds(seg["start"])
        text = seg["text"]
        if text:
            lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files to timestamped text using Groq Whisper API.",
    )
    parser.add_argument(
        "input",
        help="Path to the video or audio file to transcribe",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: same directory as input, new extension)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["md", "txt"],
        default=DEFAULT_FORMAT,
        help=f"Output format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code, e.g. 'zh', 'en', 'ja' (default: auto-detect)",
    )
    parser.add_argument(
        "-g", "--granularity",
        choices=["segment", "word"],
        default=DEFAULT_GRANULARITY,
        help=f"Timestamp granularity (default: {DEFAULT_GRANULARITY})",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=DEFAULT_CHUNK_MINUTES,
        help=f"Minutes per chunk for large files (default: {DEFAULT_CHUNK_MINUTES})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load .env from the skill directory (parent of scripts/)
    skill_dir = Path(__file__).resolve().parent.parent
    env_path = skill_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(
            f"Warning: .env file not found at {env_path}. "
            f"Copy .env.example to .env and set your GROQ_API_KEY.",
            file=sys.stderr,
        )

    # Validate API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print(
            "Error: GROQ_API_KEY is not set or is still the placeholder value.\n"
            f"Please set it in {env_path} or as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate input file
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    input_ext = Path(input_path).suffix.lower()
    all_supported = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
    if input_ext not in all_supported:
        print(
            f"Error: Unsupported file type '{input_ext}'.\n"
            f"Supported: {', '.join(sorted(all_supported))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = Path(input_path).stem
        out_ext = ".md" if args.format == "md" else ".txt"
        output_path = str(Path(input_path).parent / f"{stem}_transcript{out_ext}")

    # Initialize tools
    ffmpeg_path = _get_ffmpeg_path()
    ffprobe_path = _get_ffprobe_path()
    client = Groq(api_key=api_key)

    is_video = input_ext in VIDEO_EXTENSIONS
    file_type = "video" if is_video else "audio"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          Audio Transcription Tool                ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Input:       {input_path}")
    print(f"  Type:        {file_type}")
    print(f"  Model:       {args.model}")
    print(f"  Language:    {args.language or 'auto-detect'}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Format:      {args.format}")
    print(f"  Output:      {output_path}")
    print()

    # Process file
    segments = process_file(
        input_path=input_path,
        client=client,
        model=args.model,
        language=args.language,
        granularity=args.granularity,
        chunk_minutes=args.chunk_minutes,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
    )

    if not segments:
        print("Warning: No segments were transcribed.", file=sys.stderr)
        sys.exit(1)

    # Format output
    if args.format == "md":
        output_content = format_markdown(segments, input_path, args.model, args.language)
    else:
        output_content = format_txt(segments)

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print()
    print(f"  ✅ Transcription complete!")
    print(f"  📄 Output: {output_path}")
    print(f"  📊 Segments: {len(segments)}")


if __name__ == "__main__":
    main()
