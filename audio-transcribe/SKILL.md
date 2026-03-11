---
name: audio-transcribe
description: Transcribe video or audio files to timestamped text using Groq's Whisper Large V3 model. Supports large file chunking, auto audio extraction from video via ffmpeg, and outputs in Markdown or TXT format. Use when user asks to "transcribe audio", "transcribe video", "convert speech to text", or "generate transcript".
---

# Audio Transcribe

Transcribes video/audio files to timestamped text using Groq's Whisper Large V3 API.

## Script Directory

**Important**: All scripts are located in the `scripts/` subdirectory of this skill.

**Agent Execution Instructions**:
1. Determine this SKILL.md file's directory path as `SKILL_DIR`
2. Script path = `${SKILL_DIR}/scripts/transcribe.py`
3. Replace all `${SKILL_DIR}` in this document with the actual path

**Script Reference**:
| Script | Purpose |
|--------|---------|
| `scripts/transcribe.py` | Main CLI entry point for audio/video transcription |

## Prerequisites

- **Python 3.10+** (managed by uv)
- **Groq API Key**: Set `GROQ_API_KEY` in `${SKILL_DIR}/.env` file (see `${SKILL_DIR}/.env.example` for template)
- **ffmpeg**: Auto-detected globally; falls back to `static-ffmpeg` Python package if not found

## Setup

Before first use, install dependencies:

```bash
cd ${SKILL_DIR} && uv sync
```

## Usage

```bash
# Basic transcription (auto-detects format, outputs Markdown)
cd ${SKILL_DIR} && uv run python scripts/transcribe.py /path/to/file.mp4

# Output as TXT
cd ${SKILL_DIR} && uv run python scripts/transcribe.py /path/to/file.mp4 --format txt

# Specify language for better accuracy
cd ${SKILL_DIR} && uv run python scripts/transcribe.py /path/to/file.mp4 --language zh

# Custom output path
cd ${SKILL_DIR} && uv run python scripts/transcribe.py /path/to/file.mp4 -o /path/to/output.md

# Use word-level timestamps
cd ${SKILL_DIR} && uv run python scripts/transcribe.py /path/to/file.mp4 --granularity word
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `<input>` | | Path to video or audio file | Required |
| `--output` | `-o` | Output file path | Same dir as input, new ext |
| `--format` | `-f` | Output format: `md`, `txt` | `md` |
| `--language` | `-l` | Language code (e.g., `zh`, `en`, `ja`) | Auto-detect |
| `--granularity` | `-g` | Timestamp granularity: `segment`, `word` | `segment` |
| `--model` | `-m` | Whisper model name | `whisper-large-v3` |
| `--chunk-minutes` | | Minutes per chunk for large files | `10` |

## Supported File Types

### Video (auto-extracts audio)
`mp4`, `mkv`, `avi`, `mov`, `wmv`, `flv`, `webm`, `m4v`, `mpeg`, `mpg`, `3gp`

### Audio (direct transcription)
`mp3`, `wav`, `flac`, `m4a`, `ogg`, `aac`, `wma`, `opus`, `webm`

## Output Format

### Markdown (`--format md`)

```markdown
# Transcript: filename.mp4

- **Source**: /path/to/filename.mp4
- **Date**: 2026-03-11 14:43:07
- **Model**: whisper-large-v3
- **Language**: zh (detected)

---

## Transcript

[00:00:00] 第一段文字内容。

[00:00:15] 第二段文字内容。

[00:01:02] 第三段文字内容。
```

### TXT (`--format txt`)

```
[00:00:00] 第一段文字内容。
[00:00:15] 第二段文字内容。
[00:01:02] 第三段文字内容。
```

## Large File Handling

Files exceeding Groq's 25MB limit are automatically chunked:
1. Audio is split into 10-minute segments (configurable via `--chunk-minutes`)
2. Each segment is transcribed independently
3. Timestamps are adjusted with correct offsets
4. Results are merged into a single output file

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Required. Groq API key for Whisper transcription |

Set in `${SKILL_DIR}/.env` file. See `.env.example` for template.

## Troubleshooting

- **No GROQ_API_KEY**: Create `.env` file from `.env.example` and add your API key
- **ffmpeg not found**: The script auto-installs `static-ffmpeg` as fallback
- **File too large**: Auto-chunking handles this; adjust `--chunk-minutes` if needed
- **Poor accuracy**: Specify `--language` for better results with non-English audio
