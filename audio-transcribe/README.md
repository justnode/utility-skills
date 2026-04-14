# Audio Transcribe

音视频转录工具，支持云端（Groq Whisper API / OpenRouter Gemini 2.5 Flash）和离线（本地 NVIDIA GPU）两类转录模式。

## 功能特性

- 🎬 支持常见视频格式：mp4, mkv, avi, mov, wmv, flv, webm 等
- 🎵 支持常见音频格式：mp3, wav, flac, m4a, ogg, aac 等
- 🤖 **云端模式**：支持 Groq Whisper API 和 OpenRouter Gemini 2.5 Flash
- 💻 **离线模式**：本地 NVIDIA GPU 运行 faster-whisper，适合不上传音频内容的场景
- 📝 输出格式：Markdown、连续纯文本 TXT、SRT、VTT
- 🌍 支持自动语言检测和手动指定语言
- 🧠 支持 `--prompt` 提升专业术语识别准确率
- ✂️ 云端模式下，大于 25MB 的预处理音频会自动分片
- ⏱️ OpenRouter Gemini ASR 当前只支持段级时间戳，不支持 `word` 粒度
- 🧩 字幕默认使用 `--granularity auto`：OpenRouter 走段级时间，Groq Whisper / 本地 faster-whisper 优先用词级时间再合并成自然字幕句子

---

## 环境变量配置

### 云端模式必需

优先从全局配置文件读取：

`~/.utility-skills/.env`

```dotenv
AUDIO_TRANSCRIBE_DEFAULT_BACKEND=groq
AUDIO_TRANSCRIBE_DEFAULT_MODEL=whisper-large-v3
# 可选：如果只使用一个云服务商，可配置通用 key
# AUDIO_TRANSCRIBE_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_BASE_URL=https://api.groq.com/openai/v1
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx
# 可选
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
# OPENROUTER_HTTP_REFERER=https://your-app.example.com
# OPENROUTER_APP_TITLE=Your App Name
```

也可以继续使用当前 shell 的环境变量，且 shell 环境变量优先级更高：

```bash
# Linux/macOS
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"

# Windows (CMD)
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
set OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx

# Windows (PowerShell)
$env:GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
$env:OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"
```

获取 API Key：

- Groq: [https://console.groq.com/keys](https://console.groq.com/keys)
- OpenRouter: [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)

配置优先级：

- `AUDIO_TRANSCRIBE_DEFAULT_BACKEND` 控制不传 `--backend` 时默认使用哪个 provider
- `AUDIO_TRANSCRIBE_DEFAULT_MODEL` 控制不传 `--model` 时的默认模型
- API key 优先使用 provider 专属变量：`GROQ_API_KEY` / `OPENROUTER_API_KEY`
- 如果 provider 专属变量没配，则回退到通用 `AUDIO_TRANSCRIBE_API_KEY`

### 离线模式

离线模式不需要 API Key。

前提是本机满足：

- NVIDIA GPU
- CUDA 12.x
- 已安装 `uv sync --extra local`

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库后进入目录
cd audio-transcribe

# 仅安装云端模式依赖
uv sync

# 安装云端 + 离线 GPU 模式依赖
uv sync --extra local
```

### 2. 基本使用

```bash
# 云端转录（默认值可由 .env 配置；未配置时是 Groq + whisper-large-v3，输出 TXT 正文）
uv run python scripts/transcribe.py /path/to/video.mp4

# OpenRouter Gemini 2.5 Flash 转录
uv run python scripts/transcribe.py /path/to/video.mp4 --backend openrouter

# 离线 GPU 转录
uv run --extra local python scripts/transcribe.py /path/to/video.mp4 --backend local

# 生成中文字幕文件
uv run python scripts/transcribe.py /path/to/video.mp4 -f srt -l zh

# 字幕默认用 auto 粒度：OpenRouter 用段级时间，Groq/local 优先用词级时间
uv run python scripts/transcribe.py /path/to/video.mp4 -f srt

# 使用提示词提升专业术语识别
uv run python scripts/transcribe.py /path/to/video.mp4 --prompt "Kubernetes, Docker, DevOps"
```

### 3. 输出文件保存位置

- **默认保存位置**：输入文件所在目录
- **默认文件名**：`原文件名_transcript.扩展名`
- **自定义输出位置**：通过 `-o` 或 `--output` 指定

例如：

```bash
# 输入文件
/Users/me/Videos/lecture.mp4

# 默认输出（TXT 连续正文）
/Users/me/Videos/lecture_transcript.txt

# 指定 SRT 输出位置
uv run python scripts/transcribe.py /Users/me/Videos/lecture.mp4 \
  -f srt \
  -o /Users/me/Subtitles/lecture.zh.srt
```

---

## 在 Claude Code / Codex 中调用

### Claude Code

如果 `audio-transcribe` 目录已放在 Claude Code 可读取的位置，并且其中的 `SKILL.md` 可被读取，就可以通过自然语言让 Claude 调用它。

你可以这样说：

```
用户：请帮我转录这个视频文件 /Users/me/lecture.mp4

Claude：转录完成，输出文件：/Users/me/lecture_transcript.txt

用户：请将刚才的视频转成 SRT 字幕，语言是中文

Claude：字幕文件已生成：/Users/me/lecture_transcript.srt
```

推荐的自然语言触发方式：

- **云端模式**
  - “请用 `audio-transcribe` 把这个视频转成文字”
  - “请把 `/path/to/file.mp4` 转录成连续纯文本”
  - “请生成这个音频的中文 SRT 字幕”
  - “请使用 Groq / 云端方式转录这个文件”
  - “请用 OpenRouter 的 Gemini 2.5 Flash 转录这个音频”

- **离线模式**
  - “请离线转录这个视频，不要调用云端 API”
  - “请使用本地 GPU 版 whisper 转录这个文件”
  - “请用 faster-whisper 本地生成 VTT 字幕”

为了避免歧义，建议直接写清楚：

- **云端版**：`使用 Groq / 云端 / online`
- **OpenRouter 版**：`使用 OpenRouter / Gemini / gemini-2.5-flash`
- **离线版**：`使用本地 / 离线 / faster-whisper / local`

### OpenAI Codex

Codex 不原生支持 Claude 的 skill 格式，但可以把它当作普通命令行工具来调用。

**方式一：直接作为 CLI 工具**

在对话中告诉 Codex 脚本路径：

```
用户：请使用 /Users/me/code/audio-transcribe/scripts/transcribe.py 转录 /Users/me/video.mp4

Codex：我将执行转录命令...
      uv run --project /Users/me/code/audio-transcribe python scripts/transcribe.py /Users/me/video.mp4
```

**方式二：把使用约定写进提示词**

你可以在 Codex 的系统提示词或任务说明里明确告诉它：

- 项目目录在哪里
- 执行命令用 `uv run`
- Groq 云端模式需要 `GROQ_API_KEY`
- OpenRouter 云端模式需要 `OPENROUTER_API_KEY`
- 离线模式使用 `--backend local`

这样 Codex 更容易生成正确命令。

---

## 安装指南

### macOS

```bash
# 1. 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库并安装
git clone <repo-url> audio-transcribe
cd audio-transcribe

# 云端版本
uv sync

# 本项目离线模式按 NVIDIA GPU + CUDA 12.x 设计
# macOS 通常更适合使用云端模式
```

### Windows

```powershell
# 1. 安装 uv（PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 克隆仓库并安装
git clone <repo-url> audio-transcribe
cd audio-transcribe

# 云端版本
uv sync

# 离线版本需要 NVIDIA GPU + CUDA 12.x
# 当前离线安装说明以 Linux 为主；Windows 使用前建议先确认 CUDA 环境可用
uv sync --extra local
```

### Linux（Ubuntu/Debian 推荐）

#### 云端模式安装

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg python3 python3-pip

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. 克隆并安装
git clone <repo-url> audio-transcribe
cd audio-transcribe
uv sync

# 4. 配置 API Key
export GROQ_API_KEY="your_key_here"
```

#### 离线 GPU 模式安装

**前置要求**
- NVIDIA GPU（显存 >= 4GB 推荐）
- CUDA 12.x 运行环境可用
- NVIDIA 驱动版本 >= 525.60.13

**安装步骤**

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg python3 python3-pip

# 2. 验证 GPU / CUDA 可用
nvidia-smi
# 应显示 CUDA Version: 12.x

# 3. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 4. 克隆仓库
git clone <repo-url> audio-transcribe
cd audio-transcribe

# 5. 安装云端 + 离线依赖（关键步骤）
# 这会安装 faster-whisper + nvidia-cublas-cu12 + nvidia-cudnn-cu12
uv sync --extra local

# 6. 验证 CLI 是否可用
uv run --extra local python scripts/transcribe.py --help

# 7. 实际测试一次离线 GPU 转录
uv run --extra local python scripts/transcribe.py /path/to/test.mp3 \
  --backend local \
  --model base
```

**常见问题**

1. **cuDNN 库加载失败**
   ```bash
   # 手动设置 LD_LIBRARY_PATH（脚本已自动处理，但如需手动）
   export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib, nvidia.cudnn.lib, os; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH
   ```

2. **CUDA 版本不匹配**
   ```bash
   # 检查 CUDA 版本
   nvidia-smi
   # 本项目当前离线模式仅面向 CUDA 12.x
   # 如果不是 CUDA 12.x，建议改用云端模式，或先升级本机 CUDA / 驱动环境
   ```

3. **模型下载慢（首次运行）**
   ```bash
   # 设置 HuggingFace 镜像（可选）
   export HF_ENDPOINT=https://hf-mirror.com
   ```

---

## 命令行参数

```
usage: transcribe.py [-h] [-o OUTPUT] [-f {md,txt,srt,vtt}]
                     [-l LANGUAGE] [-g {auto,segment,word}] [-m MODEL] [-p PROMPT]
                     [--chunk-minutes CHUNK_MINUTES]
                     [-b {groq,openrouter,local}]
                     [--compute-type COMPUTE_TYPE]
                     input

参数：
  input                 输入的音视频文件路径
  -o, --output         输出文件路径（默认：输入文件目录 + _transcript.格式）
  -f, --format         输出格式：txt（默认）、md、srt、vtt
  -l, --language       语言代码：zh、en、ja 等（默认：自动检测）
  -g, --granularity    时间戳粒度：auto（默认）、segment（段落）、word（单词级）
  -m, --model          模型名称（若设置了 AUDIO_TRANSCRIBE_DEFAULT_MODEL，则优先使用该值；否则使用所选 backend 的默认模型：groq=whisper-large-v3，openrouter=google/gemini-2.5-flash；离线常用：large-v3、base、small、medium）
  -p, --prompt         提示词，帮助识别专业术语、人名等
  --chunk-minutes      大文件分片时长，默认 10 分钟（仅云端模式）
  -b, --backend        后端：groq（云端默认）、openrouter（Gemini ASR）、local（离线 GPU）
  --compute-type       本地精度：float16（默认）、float32、int8_float16
```

---

## 使用示例

### 云端模式

```bash
# 基础转录
uv run python scripts/transcribe.py ~/Downloads/interview.mp4

# 生成中文字幕
uv run python scripts/transcribe.py ~/Downloads/movie.mp4 -f srt -l zh -o ~/Downloads/movie.zh.srt

# 快速转录（turbo 模型）
uv run python scripts/transcribe.py ~/Downloads/podcast.mp3 --model whisper-large-v3-turbo

# 技术会议转录（使用提示词）
uv run python scripts/transcribe.py ~/Downloads/meeting.mp4 \
  --prompt "AWS, EC2, S3, Lambda, Serverless, Kubernetes" \
  -l en -f md
```

### 离线 GPU 模式

```bash
# 基础离线转录
uv run --extra local python scripts/transcribe.py ~/Downloads/video.mp4 --backend local

# 使用 large-v3 模型（高质量，慢速）
uv run --extra local python scripts/transcribe.py ~/Downloads/video.mp4 \
  --backend local -m large-v3

# 使用 base 模型（快速，适合短音频）
uv run --extra local python scripts/transcribe.py ~/Downloads/short.mp3 \
  --backend local -m base --compute-type int8_float16

# 生成 WebVTT 字幕
uv run --extra local python scripts/transcribe.py ~/Downloads/lecture.mp4 \
  --backend local -f vtt -l en
```

---

## 模式选择建议

- **优先选 Groq 云端**
  - 你希望配置最少
  - 你已经有 `GROQ_API_KEY`
  - 你更看重接入简单和转录速度

- **优先选本地离线**
  - 你不希望音频上传到云端
  - 你有可用的 NVIDIA GPU
  - 你愿意准备 CUDA 12.x 本地环境

---

## 项目结构

```
audio-transcribe/
├── SKILL.md              # Claude/Codex 技能定义文件
├── README.md             # 本文件
├── pyproject.toml        # 项目配置和依赖
├── uv.lock              # uv 锁定文件
├── scripts/
│   ├── __init__.py
│   └── transcribe.py     # 主转录脚本
└── .gitignore
```

---

## 注意事项

1. **首次运行离线模式**：会自动从 HuggingFace 下载模型文件，具体大小取决于模型规模，请确保网络畅通。
2. **模型缓存位置**：`~/.cache/huggingface/hub/`，可手动清理。
3. **大文件处理**：云端模式 >25MB 自动分片，离线模式无大小限制。
4. **隐私安全**：离线模式所有数据在本地处理，不上传云端。

---

## 故障排除

### 问题：无法加载 CUDA 库

```
RuntimeError: CUDA failed with error CUDA driver version is insufficient
```

**解决**：
```bash
# 检查驱动版本
nvidia-smi
# 确保驱动 >= 525.60.13，CUDA >= 12.0

# 更新 NVIDIA 驱动（Ubuntu）
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

### 问题：cuDNN 加载失败

```
Could not load library libcudnn*.so
```

**解决**：
```bash
# 确保使用 uv sync --extra local 安装
# 脚本会自动设置 LD_LIBRARY_PATH，如需手动：
export LD_LIBRARY_PATH=$(python3 -c "import nvidia.cublas.lib, nvidia.cudnn.lib, os; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))"):$LD_LIBRARY_PATH
```

### 问题：离线模式启动后仍然报 GPU / CUDA 不可用

**排查顺序**：

```bash
# 1. 先确认驱动和 GPU 可见
nvidia-smi

# 2. 确认已安装离线依赖
uv sync --extra local

# 3. 再执行离线命令
uv run --extra local python scripts/transcribe.py /path/to/test.mp3 --backend local
```

### 问题：Groq API 报错

```
Error: GROQ_API_KEY environment variable is not set
```

**解决**：
```bash
export GROQ_API_KEY="gsk_xxxxx"
# 或添加到 ~/.bashrc / ~/.zshrc 永久生效
```

---

## 许可证

MIT License
