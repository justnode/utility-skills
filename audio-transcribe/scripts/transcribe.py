#!/usr/bin/env python3
"""
Audio/Video Transcription Script.

Transcribes video or audio files to timestamped text.
- Backends: Groq Whisper API, OpenRouter Gemini 2.5 Flash, or faster-whisper.
- Automatically extracts audio from video files using ffmpeg.
- Supports large file chunking with configurable chunk duration.
- Outputs in Markdown, TXT, SRT, or VTT format with timestamps.
"""

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib import error as urllib_error
from urllib import request as urllib_request

# groq and faster_whisper are imported lazily based on --backend

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

# Prepared-audio chunk threshold for cloud backends (bytes)
CLOUD_CHUNK_THRESHOLD_BYTES = 25 * 1024 * 1024  # 25 MB

DEFAULT_CHUNK_MINUTES = 10
DEFAULT_FORMAT = "txt"
DEFAULT_GRANULARITY = "auto"

# API retry settings (cloud backends)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Backend defaults
DEFAULT_BACKEND = "groq"
DEFAULT_COMPUTE_TYPE = "float16"
GLOBAL_ENV_PATH = Path.home() / ".utility-skills" / ".env"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BACKEND_ENV_VAR = "AUDIO_TRANSCRIBE_DEFAULT_BACKEND"
DEFAULT_MODEL_ENV_VAR = "AUDIO_TRANSCRIBE_DEFAULT_MODEL"
DEFAULT_API_KEY_ENV_VAR = "AUDIO_TRANSCRIBE_API_KEY"

BACKEND_DEFAULT_MODELS = {
    "groq": "whisper-large-v3",
    "openrouter": "google/gemini-2.5-flash",
    "local": "whisper-large-v3",
}


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


TranscribeFn = Callable[[str], dict]

SUBTITLE_MAX_CHARS = 22
SUBTITLE_MAX_DURATION_SECONDS = 4.5
SUBTITLE_MAX_GAP_SECONDS = 0.6
SENTENCE_BREAK_PUNCTUATION = set("。！？!?；;")
SOFT_BREAK_PUNCTUATION = set("，、,")


def resolve_default_backend() -> str:
    """Resolve the default backend from environment configuration."""
    configured_backend = os.environ.get(DEFAULT_BACKEND_ENV_VAR, "").strip()
    if not configured_backend:
        return DEFAULT_BACKEND

    if configured_backend not in BACKEND_DEFAULT_MODELS:
        supported = ", ".join(BACKEND_DEFAULT_MODELS)
        print(
            f"Warning: {DEFAULT_BACKEND_ENV_VAR}={configured_backend!r} is not supported. "
            f"Expected one of: {supported}. Falling back to {DEFAULT_BACKEND}.",
            file=sys.stderr,
        )
        return DEFAULT_BACKEND

    return configured_backend


def resolve_default_model() -> str | None:
    """Resolve the optional global default model from environment configuration."""
    configured_model = os.environ.get(DEFAULT_MODEL_ENV_VAR, "").strip()
    if configured_model:
        return configured_model
    return None


def resolve_api_key(*env_var_names: str) -> str | None:
    """Return the first configured API key from the provided env vars."""
    for env_var_name in env_var_names:
        value = os.environ.get(env_var_name, "").strip()
        if value:
            return value
    return None


def is_cjk_character(char: str) -> bool:
    """Return True when a character belongs to a common CJK Unicode block."""
    if not char:
        return False
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def merge_word_token(existing_text: str, token: str) -> str:
    """Append a word token using spacing rules that work for CJK and Latin text."""
    token = token.strip()
    if not token:
        return existing_text
    if not existing_text:
        return token

    if token[0] in SENTENCE_BREAK_PUNCTUATION or token[0] in SOFT_BREAK_PUNCTUATION:
        return existing_text + token

    if is_cjk_character(existing_text[-1]) or is_cjk_character(token[0]):
        return existing_text + token

    return existing_text + " " + token


def resolve_effective_granularity(backend: str, model: str, granularity: str, output_format: str) -> str:
    """Resolve the timestamp source used for a given output."""
    if granularity != "auto":
        return granularity

    if output_format in {"srt", "vtt"}:
        if backend == "openrouter":
            return "segment"
        if backend == "local":
            return "word"
        if backend == "groq" and "whisper-large-v3" in model:
            return "word"

    return "segment"


def build_subtitle_segments_from_words(words: list[dict]) -> list[dict]:
    """Group word-level timestamps into readable subtitle segments."""
    subtitle_segments: list[dict] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None
    previous_end: float | None = None

    def flush_current() -> None:
        nonlocal current_text, current_start, current_end
        if current_text and current_start is not None and current_end is not None:
            subtitle_segments.append({
                "start": current_start,
                "end": current_end,
                "text": current_text.strip(),
            })
        current_text = ""
        current_start = None
        current_end = None

    for word in words:
        token = str(word.get("word", "") or "").strip()
        start = float(word.get("start", 0) or 0)
        end = float(word.get("end", 0) or 0)
        if not token:
            continue

        if (
            current_text
            and previous_end is not None
            and start - previous_end >= SUBTITLE_MAX_GAP_SECONDS
        ):
            flush_current()

        if current_start is None:
            current_start = start

        candidate_text = merge_word_token(current_text, token)
        current_text = candidate_text
        current_end = end
        previous_end = end

        duration = (current_end - current_start) if current_start is not None else 0
        last_char = current_text[-1] if current_text else ""

        should_flush = False
        if len(current_text) >= SUBTITLE_MAX_CHARS:
            should_flush = True
        elif duration >= SUBTITLE_MAX_DURATION_SECONDS:
            should_flush = True
        elif last_char in SENTENCE_BREAK_PUNCTUATION:
            should_flush = True
        elif last_char in SOFT_BREAK_PUNCTUATION and len(current_text) >= 10:
            should_flush = True

        if should_flush:
            flush_current()

    flush_current()
    return subtitle_segments


def build_plain_text(transcript_data: dict) -> str:
    """Build a continuous plain-text transcript without timestamps."""
    segments = transcript_data.get("segments", []) or []
    if segments:
        text = ""
        for segment in segments:
            piece = str(segment.get("text", "") or "").strip()
            if not piece:
                continue
            text = merge_word_token(text, piece)
        if text:
            return text.strip()

    return str(transcript_data.get("text", "") or "").strip()


def build_output_segments(data: dict, granularity: str, output_format: str) -> list[dict]:
    """Build output segments using the timestamp granularity requested upstream."""
    if output_format in {"srt", "vtt"} and granularity == "word":
        words = data.get("words", []) or []
        if words:
            return build_subtitle_segments_from_words(words)
        if data.get("segments"):
            return _extract_segments(data, "segment", offset=0.0)

    return _extract_segments(data, granularity, offset=0.0)


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
# Transcription backends
# ---------------------------------------------------------------------------


def transcribe_groq(
    client,
    audio_path: str,
    model: str,
    language: str | None,
    granularity: str,
    prompt: str | None = None,
) -> dict:
    """Transcribe a single audio file via Groq API and return the raw response.

    Retries up to MAX_RETRIES times on transient API errors.
    """
    kwargs: dict = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": [granularity],
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(audio_path, "rb") as f:
                kwargs["file"] = f
                response = client.audio.transcriptions.create(**kwargs)

            # The SDK returns a Pydantic model; convert to dict for uniform handling
            if hasattr(response, "model_dump"):
                return response.model_dump()
            return json.loads(json.dumps(response, default=str))
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


def _strip_json_fences(content: str) -> str:
    """Remove optional markdown code fences around JSON responses."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_openrouter_message_content(response_data: dict) -> str:
    """Extract the assistant message content from an OpenRouter response."""
    choices = response_data.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response did not include any choices")

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)

    raise TypeError(f"Unexpected OpenRouter content type: {type(content)}")


def _normalize_transcript_result(result: dict) -> dict:
    """Normalize a provider response into the shared transcript structure."""
    normalized = {
        "text": str(result.get("text", "") or "").strip(),
        "language": result.get("language"),
        "segments": [],
        "words": [],
    }

    for segment in result.get("segments", []) or []:
        normalized["segments"].append({
            "start": float(segment.get("start", 0) or 0),
            "end": float(segment.get("end", 0) or 0),
            "text": str(segment.get("text", "") or "").strip(),
        })

    for word in result.get("words", []) or []:
        normalized["words"].append({
            "start": float(word.get("start", 0) or 0),
            "end": float(word.get("end", 0) or 0),
            "word": str(word.get("word", "") or "").strip(),
        })

    return normalized


def transcribe_openrouter(
    api_key: str,
    base_url: str,
    audio_path: str,
    model: str,
    language: str | None,
    prompt: str | None = None,
    referer: str | None = None,
    title: str | None = None,
) -> dict:
    """Transcribe a single audio file via OpenRouter chat completions."""
    audio_bytes = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    instruction_lines = [
        "Transcribe this audio clip faithfully.",
        "Return JSON only.",
        "Provide top-level keys: text, language, segments, words.",
        "Each segment must include start, end, and text in seconds relative to this clip.",
        "Use best-effort segment timestamps and preserve the spoken wording.",
        "Leave words as an empty array.",
        "Do not summarize or translate unless the audio itself does so.",
    ]
    if language:
        instruction_lines.append(
            f"The expected spoken language is {language}; keep that language in the transcript when possible."
        )
    if prompt:
        instruction_lines.append(
            f"Domain-specific hint for names/terms: {prompt}"
        )

    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "transcript_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "language": {"type": ["string", "null"]},
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "text": {"type": "string"},
                                },
                                "required": ["start", "end", "text"],
                            },
                        },
                        "words": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"},
                                    "word": {"type": "string"},
                                },
                                "required": ["start", "end", "word"],
                            },
                        },
                    },
                    "required": ["text", "language", "segments", "words"],
                },
            },
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n".join(instruction_lines),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "flac",
                        },
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib_request.Request(
                base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib_request.urlopen(req) as response:
                response_data = json.loads(response.read().decode("utf-8"))

            content = _extract_openrouter_message_content(response_data)
            parsed = json.loads(_strip_json_fences(content))
            return _normalize_transcript_result(parsed)
        except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * attempt
                print(f"  ⚠ API error (attempt {attempt}/{MAX_RETRIES}): {exc}")
                print(f"    Retrying in {delay}s …")
                time.sleep(delay)
            else:
                break

    print(f"  ✗ API failed after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)
    raise last_error


def transcribe_local(
    fw_model,
    audio_path: str,
    language: str | None,
    granularity: str,
    prompt: str | None = None,
) -> dict:
    """Transcribe a single audio file via faster-whisper locally.

    Returns a dict matching Groq's verbose_json structure for uniform handling.
    """
    kwargs: dict = {
        "beam_size": 5,
        "word_timestamps": granularity == "word",
        "vad_filter": True,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt

    segments_gen, info = fw_model.transcribe(audio_path, **kwargs)
    segments_list = list(segments_gen)

    # Build a dict compatible with Groq's verbose_json response
    result: dict = {
        "text": " ".join(seg.text.strip() for seg in segments_list),
        "language": info.language,
        "segments": [],
        "words": [],
    }

    for seg in segments_list:
        result["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        if seg.words:
            for w in seg.words:
                result["words"].append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word.strip(),
                })

    return result


def resolve_model(
    backend: str,
    requested_model: str | None,
    configured_default_model: str | None = None,
) -> str:
    """Resolve the effective model for a backend."""
    if requested_model:
        return requested_model
    if configured_default_model:
        return configured_default_model
    return BACKEND_DEFAULT_MODELS[backend]


def offset_transcript_data(data: dict, offset: float) -> dict:
    """Offset transcript timestamps by a chunk start time."""
    normalized = _normalize_transcript_result(data)
    for segment in normalized["segments"]:
        segment["start"] += offset
        segment["end"] += offset
    for word in normalized["words"]:
        word["start"] += offset
        word["end"] += offset
    return normalized


def merge_transcript_data(results: list[dict]) -> dict:
    """Merge multiple chunk transcript results into one transcript structure."""
    merged = {
        "text": "",
        "language": None,
        "segments": [],
        "words": [],
    }

    text_parts: list[str] = []
    for result in results:
        normalized = _normalize_transcript_result(result)
        if not merged["language"] and normalized.get("language"):
            merged["language"] = normalized["language"]
        if normalized["text"]:
            text_parts.append(normalized["text"])
        merged["segments"].extend(normalized["segments"])
        merged["words"].extend(normalized["words"])

    merged["text"] = " ".join(part for part in text_parts if part).strip()
    return merged


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def process_file(
    input_path: str,
    transcribe_fn,
    chunk_minutes: int,
    ffmpeg_path: str,
    ffprobe_path: str,
    need_chunking: bool = True,
) -> dict:
    """
    Process an input file: extract audio if needed, chunk if large, transcribe.

    Args:
        transcribe_fn: Callable(audio_path) -> dict  (shared transcript dict)
        need_chunking: If True, chunk prepared audio files exceeding the cloud
                       threshold. If False, process as a single file.

    Returns a transcript dict with keys: text, language, segments, words.
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
        if not need_chunking or file_size <= CLOUD_CHUNK_THRESHOLD_BYTES:
            print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
            print(f"  Transcribing …")
            result = transcribe_fn(prepared_audio)
            return _normalize_transcript_result(result)
        else:
            print(
                f"  File size: {file_size / 1024 / 1024:.1f} MB "
                f"(exceeds {CLOUD_CHUNK_THRESHOLD_BYTES / 1024 / 1024:.0f} MB cloud chunk threshold)"
            )
            duration = get_audio_duration(prepared_audio, ffprobe_path)
            chunk_duration = chunk_minutes * 60
            num_chunks = math.ceil(duration / chunk_duration)
            print(f"  Duration: {_fmt_seconds(duration)} | Splitting into {num_chunks} chunk(s) …")

            all_results: list[dict] = []

            for i in range(num_chunks):
                start = i * chunk_duration
                chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.flac")

                # Determine actual chunk length (last chunk may be shorter)
                remaining = duration - start
                actual_duration = min(chunk_duration, remaining)

                print(f"  Chunk {i + 1}/{num_chunks}: {_fmt_seconds(start)} – {_fmt_seconds(start + actual_duration)}")
                extract_audio(prepared_audio, chunk_path, ffmpeg_path, start_seconds=start, duration_seconds=actual_duration)

                result = transcribe_fn(chunk_path)
                all_results.append(offset_transcript_data(result, offset=start))

            return merge_transcript_data(all_results)


def _extract_segments(data: dict, granularity: str, offset: float) -> list[dict]:
    """Extract timestamped segments from the shared transcript structure."""
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


def format_txt(transcript_data: dict) -> str:
    """Format transcription as continuous plain text without timestamps."""
    return build_plain_text(transcript_data)


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    """Format seconds into VTT timestamp: HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_srt(segments: list[dict]) -> str:
    """Format transcription segments as SRT subtitle file."""
    lines = []
    idx = 1
    for seg in segments:
        text = seg["text"]
        if not text:
            continue
        start = _fmt_srt_time(seg["start"])
        end = _fmt_srt_time(seg["end"])
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def format_vtt(segments: list[dict]) -> str:
    """Format transcription segments as WebVTT subtitle file."""
    lines = ["WEBVTT", ""]
    idx = 1
    for seg in segments:
        text = seg["text"]
        if not text:
            continue
        start = _fmt_vtt_time(seg["start"])
        end = _fmt_vtt_time(seg["end"])
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser(
    default_backend: str,
    configured_default_model: str | None,
) -> argparse.ArgumentParser:
    if configured_default_model:
        model_help = (
            "Model name. Defaults to "
            f"{DEFAULT_MODEL_ENV_VAR}={configured_default_model}; "
            "otherwise uses the selected backend default."
        )
    else:
        model_help = (
            "Model name. Defaults to the selected backend default "
            f"(groq: {BACKEND_DEFAULT_MODELS['groq']}, "
            f"openrouter: {BACKEND_DEFAULT_MODELS['openrouter']}, "
            f"local: {BACKEND_DEFAULT_MODELS['local']})."
        )

    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files to timestamped text. "
                    "Supports Groq Whisper API, OpenRouter Gemini 2.5 Flash, "
                    "and faster-whisper (local/offline).",
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
        choices=["md", "txt", "srt", "vtt"],
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
        choices=["auto", "segment", "word"],
        default=DEFAULT_GRANULARITY,
        help=(
            "Timestamp granularity. 'auto' uses segment timestamps by default, "
            "and for subtitles prefers word timings on Groq whisper-large-v3 and local faster-whisper."
        ),
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=model_help,
    )
    parser.add_argument(
        "-p", "--prompt",
        default=None,
        help="Prompt hint for Whisper (e.g. proper nouns, technical terms) to improve accuracy",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=DEFAULT_CHUNK_MINUTES,
        help=f"Minutes per chunk for large files (default: {DEFAULT_CHUNK_MINUTES})",
    )
    parser.add_argument(
        "-b", "--backend",
        choices=["groq", "openrouter", "local"],
        default=default_backend,
        help=f"Transcription backend (default: {default_backend})",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"Compute type for local backend: 'float16', 'float32', 'int8_float16' (default: {DEFAULT_COMPUTE_TYPE})",
    )
    return parser


def _setup_cuda_lib_path() -> None:
    """Ensure LD_LIBRARY_PATH includes nvidia-cublas and nvidia-cudnn installed by pip/uv."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        cublas_dir = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_dir = os.path.dirname(nvidia.cudnn.lib.__file__)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [cublas_dir, cudnn_dir]
        if existing:
            paths.append(existing)
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
    except ImportError:
        pass  # Libraries might be installed system-wide


def setup_groq_backend(
    model: str,
    language: str | None,
    granularity: str,
    prompt: str | None,
) -> tuple[TranscribeFn, bool, str]:
    """Create a Groq transcription backend."""
    from groq import Groq

    api_key = resolve_api_key("GROQ_API_KEY", DEFAULT_API_KEY_ENV_VAR)
    if not api_key or api_key == "your_groq_api_key_here":
        print(
            "Error: GROQ_API_KEY environment variable is not set.\n"
            f"Please add it to {GLOBAL_ENV_PATH}, for example:\n"
            f"{DEFAULT_BACKEND_ENV_VAR}=groq\n"
            f"{DEFAULT_MODEL_ENV_VAR}=whisper-large-v3\n"
            "GROQ_API_KEY=gsk_...\n"
            f"{DEFAULT_API_KEY_ENV_VAR}=gsk_...  # optional generic fallback\n"
            "Get a key from: https://console.groq.com/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    client_kwargs = {"api_key": api_key}
    groq_base_url = os.environ.get("GROQ_BASE_URL", "").strip()
    if groq_base_url:
        client_kwargs["base_url"] = groq_base_url

    client = Groq(**client_kwargs)

    def transcribe_fn(audio_path: str) -> dict:
        return transcribe_groq(client, audio_path, model, language, granularity, prompt)

    return transcribe_fn, True, f"groq ({model})"


def setup_openrouter_backend(
    model: str,
    language: str | None,
    granularity: str,
    prompt: str | None,
) -> tuple[TranscribeFn, bool, str]:
    """Create an OpenRouter transcription backend."""
    api_key = resolve_api_key("OPENROUTER_API_KEY", DEFAULT_API_KEY_ENV_VAR)
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY environment variable is not set.\n"
            f"Please add it to {GLOBAL_ENV_PATH}, for example:\n"
            f"{DEFAULT_BACKEND_ENV_VAR}=openrouter\n"
            f"{DEFAULT_MODEL_ENV_VAR}=google/gemini-2.5-flash\n"
            "OPENROUTER_API_KEY=sk-or-...\n"
            f"{DEFAULT_API_KEY_ENV_VAR}=sk-or-...  # optional generic fallback\n"
            "Get a key from: https://openrouter.ai/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    if granularity == "word":
        print(
            "Error: --granularity word is not supported for the openrouter backend.\n"
            "Use --granularity segment for OpenRouter Gemini ASR.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_OPENROUTER_BASE_URL
    referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip() or None
    title = os.environ.get("OPENROUTER_APP_TITLE", "").strip() or None

    def transcribe_fn(audio_path: str) -> dict:
        return transcribe_openrouter(
            api_key=api_key,
            base_url=base_url,
            audio_path=audio_path,
            model=model,
            language=language,
            prompt=prompt,
            referer=referer,
            title=title,
        )

    return transcribe_fn, True, f"openrouter ({model})"


def setup_local_backend(
    model: str,
    language: str | None,
    granularity: str,
    prompt: str | None,
    compute_type: str,
) -> tuple[TranscribeFn, bool, str]:
    """Create a local faster-whisper backend."""
    _setup_cuda_lib_path()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper is not installed.\n"
            "Install it: uv sync --project <SKILL_DIR> --extra local",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loading model '{model}' on cuda ({compute_type}) …")
    fw_model = WhisperModel(model, device="cuda", compute_type=compute_type)

    def transcribe_fn(audio_path: str) -> dict:
        return transcribe_local(fw_model, audio_path, language, granularity, prompt)

    return transcribe_fn, False, f"local ({model}, cuda, {compute_type})"


def main() -> None:
    # Load shared user-level configuration before validating backend credentials.
    load_env_file(GLOBAL_ENV_PATH)
    default_backend = resolve_default_backend()
    configured_default_model = resolve_default_model()
    parser = build_parser(default_backend, configured_default_model)
    args = parser.parse_args()
    resolved_model = resolve_model(args.backend, args.model, configured_default_model)
    effective_granularity = resolve_effective_granularity(
        args.backend, resolved_model, args.granularity, args.format
    )

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
    format_ext_map = {"md": ".md", "txt": ".txt", "srt": ".srt", "vtt": ".vtt"}
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = Path(input_path).stem
        out_ext = format_ext_map[args.format]
        output_path = str(Path(input_path).parent / f"{stem}_transcript{out_ext}")

    # Initialize ffmpeg tools
    ffmpeg_path = _get_ffmpeg_path()
    ffprobe_path = _get_ffprobe_path()

    is_video = input_ext in VIDEO_EXTENSIONS
    file_type = "video" if is_video else "audio"

    # Initialize backend
    if args.backend == "groq":
        transcribe_fn, need_chunking, backend_display = setup_groq_backend(
            resolved_model, args.language, effective_granularity, args.prompt
        )
    elif args.backend == "openrouter":
        transcribe_fn, need_chunking, backend_display = setup_openrouter_backend(
            resolved_model, args.language, effective_granularity, args.prompt
        )
    else:
        transcribe_fn, need_chunking, backend_display = setup_local_backend(
            resolved_model, args.language, effective_granularity, args.prompt, args.compute_type
        )

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║          Audio Transcription Tool                ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Input:       {input_path}")
    print(f"  Type:        {file_type}")
    print(f"  Backend:     {backend_display}")
    print(f"  Language:    {args.language or 'auto-detect'}")
    print(f"  Granularity: {args.granularity}")
    if args.format in {"srt", "vtt"}:
        print(f"  Subtitle TS: {effective_granularity}")
    print(f"  Format:      {args.format}")
    if args.prompt:
        print(f"  Prompt:      {args.prompt}")
    print(f"  Output:      {output_path}")
    print()

    # Process file
    transcript_data = process_file(
        input_path=input_path,
        transcribe_fn=transcribe_fn,
        chunk_minutes=args.chunk_minutes,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        need_chunking=need_chunking,
    )

    segments = build_output_segments(transcript_data, effective_granularity, args.format)

    if not segments:
        print("Warning: No segments were transcribed.", file=sys.stderr)
        sys.exit(1)

    # Format output
    formatters = {
        "md": lambda: format_markdown(segments, input_path, resolved_model, args.language),
        "txt": lambda: format_txt(transcript_data),
        "srt": lambda: format_srt(segments),
        "vtt": lambda: format_vtt(segments),
    }
    output_content = formatters[args.format]()

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
