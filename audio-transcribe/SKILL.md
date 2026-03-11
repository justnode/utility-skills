---
name: audio-transcribe
description: >
  Transcribe video or audio files to timestamped text using Groq Whisper API.
  Supports all common video formats (mp4, mkv, avi, mov, etc.) and audio formats
  (mp3, wav, flac, m4a, etc.). Features: auto audio extraction from video via
  ffmpeg, large file chunking (over 25MB), multiple output formats (Markdown, TXT,
  SRT, VTT subtitles), language specification, word-level timestamps, and prompt
  hints for domain-specific accuracy. Use when user asks to: "transcribe audio",
  "transcribe video", "convert speech to text", "generate transcript",
  "extract text from video", "generate subtitles", "create SRT/VTT",
  "语音转文字", "提取音频文案", "视频转文字", "生成字幕".
---

# Audio Transcribe

Transcribe video/audio files to timestamped text via Groq Whisper API.

## Setup

1. Determine this SKILL.md file's directory path as `SKILL_DIR`
2. Script path: `${SKILL_DIR}/scripts/transcribe.py`
3. Require environment variable `GROQ_API_KEY` (obtain from https://console.groq.com/keys)
4. ffmpeg: auto-detected globally; falls back to `static-ffmpeg` Python package

If `GROQ_API_KEY` is not set, prompt the user to set it before running.

## Workflow

1. Determine the input file type and desired output format
2. Choose the appropriate command options:
   - **Subtitles needed?** → use `--format srt` or `--format vtt`
   - **Non-English audio?** → add `--language zh` (or `en`, `ja`, etc.)
   - **Domain-specific terms?** → add `--prompt "术语1, 术语2"` to improve accuracy
   - **Fast transcription preferred?** → add `--model whisper-large-v3-turbo`
3. Run the script and deliver the output file to the user

## Usage

```bash
# Basic transcription (outputs Markdown with timestamps)
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4

# Generate SRT subtitles in Chinese
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 -f srt -l zh

# Fast transcription with domain prompt
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 --model whisper-large-v3-turbo --prompt "React, TypeScript, Vite"

# Custom output path, VTT format
uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py /path/to/file.mp4 -f vtt -o /path/to/output.vtt
```

Run `uv run --project ${SKILL_DIR} python ${SKILL_DIR}/scripts/transcribe.py --help` for all options.

## Key Details

- Files >25MB are auto-chunked (configurable via `--chunk-minutes`, default 10)
- Output defaults to same directory as input with appropriate extension
- All video formats auto-extract audio; all audio formats accepted directly
- The `--prompt` option provides context hints to Whisper (e.g., proper nouns, technical terms) — it does NOT instruct the model, just guides recognition accuracy
