---
name: audio-transcribe
description: >
  Transcribe video or audio files to timestamped text. Two backends:
  Groq Whisper API (online, fast, requires API key) and faster-whisper
  (local/offline, GPU with CUDA 12.x, no API key needed). Supports all
  common video formats (mp4, mkv, avi, mov, etc.) and audio formats
  (mp3, wav, flac, m4a, etc.). Features: auto audio extraction from video
  via ffmpeg, large file chunking, multiple output formats (Markdown, TXT,
  SRT, VTT subtitles), language specification, word-level timestamps, VAD
  filtering (local), and prompt hints for domain-specific accuracy.
  Use when user asks to: "transcribe audio", "transcribe video",
  "convert speech to text", "generate transcript", "extract text from video",
  "generate subtitles", "create SRT/VTT", "offline transcription",
  "local whisper", "语音转文字", "提取音频文案", "视频转文字", "生成字幕".
---

# Audio Transcribe

Transcribe video/audio files to timestamped text. Supports online (Groq API) and offline (faster-whisper GPU) backends.

## Setup

1. Determine this SKILL.md file's directory path as `SKILL_DIR`
2. Script path: `${SKILL_DIR}/scripts/transcribe.py`
3. ffmpeg: auto-detected globally; falls back to `static-ffmpeg` Python package
4. Backend prerequisites:
   - **Groq (default)**: Require environment variable `GROQ_API_KEY` (obtain from https://console.groq.com/keys). If not set, prompt the user to set it before running.
   - **Local (GPU)**: Requires NVIDIA GPU with CUDA 12.x. Install optional dependency first: `uv sync --project ${SKILL_DIR} --extra local` (includes faster-whisper, nvidia-cublas-cu12, nvidia-cudnn-cu12). No API key needed.

## Workflow

1. Determine the input file type and desired output format
2. Choose the backend:
   - **Online (default)** → `--backend groq` (requires `GROQ_API_KEY`)
   - **Offline / no API key** → `--backend local` (requires NVIDIA GPU + CUDA 12.x)
3. Choose the appropriate command options:
   - **Subtitles needed?** → use `--format srt` or `--format vtt`
   - **Non-English audio?** → add `--language zh` (or `en`, `ja`, etc.)
   - **Domain-specific terms?** → add `--prompt "术语1, 术语2"` to improve accuracy
   - **Fast online transcription?** → add `--model whisper-large-v3-turbo`
   - **Local precision control?** → add `--compute-type float16` (default) or `int8_float16`
4. Run the script and deliver the output file to the user

## Usage

```bash
# Online: basic transcription (Groq API, outputs Markdown)
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4

# Online: SRT subtitles in Chinese
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 -f srt -l zh

# Offline: local GPU transcription (auto model download on first run)
uv run --project ${SKILL_DIR} --extra local python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 -b local

# Offline: specific model, VTT output
uv run --project ${SKILL_DIR} --extra local python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 -b local -m large-v3 -f vtt

# Domain prompt (works with both backends)
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 --prompt "React, TypeScript, Vite"
```

Run `uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py --help` for all options.

## Key Details

- **Groq backend**: files over 25MB are auto-chunked (configurable via `--chunk-minutes`, default 10)
- **Local backend**: requires NVIDIA GPU + CUDA 12.x; no file size limit; includes VAD filtering automatically
- CUDA libraries (cuBLAS, cuDNN 9) are auto-installed via `--extra local`; `LD_LIBRARY_PATH` is set automatically at runtime
- Output defaults to same directory as input with appropriate extension
- All video formats auto-extract audio; all audio formats accepted directly
- The `--prompt` option provides context hints to Whisper (proper nouns, technical terms) — it does NOT instruct the model, just guides recognition accuracy
- Local model is downloaded automatically on first use (cached in `~/.cache/huggingface/`)
