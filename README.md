# rust-whisper-mcp

Local MCP server: Whisper large-v3 on a dedicated NVIDIA GPU via Docker, exposing two tools — `transcribe_file` (local media) and `transcribe_youtube` (any yt-dlp URL). Written in Rust (rmcp + whisper-rs CUDA).

## Stack

- Rust + [`rmcp`](https://docs.rs/rmcp) 1.5 — MCP server, Streamable HTTP transport
- [`whisper-rs`](https://docs.rs/whisper-rs) 0.13 with `cuda` feature — GGML large-v3 on GPU
- `ffmpeg` (any → 16kHz mono PCM) + `yt-dlp` (YouTube → wav)
- Docker Compose, CUDA 12.4 runtime, GPU device 1 only

## Pre-flight (run BEFORE first build)

In PowerShell on the host, verify GPU2 is reachable from Docker:

```powershell
docker run --rm --gpus '"device=1"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You must see your GPU2 listed. If not — install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and enable GPU support in Docker Desktop → Settings → Resources → WSL Integration / GPU.

## Build & run

```powershell
# from C:\dev\lab\rust-whisper-mcp
docker compose build
docker compose up -d
docker compose logs -f whisper-mcp
```

First start downloads `ggml-large-v3.bin` (~3 GB) into the `whisper-models` volume. Subsequent starts skip that.

Health: `curl http://localhost:17800/health` → `ok`
MCP endpoint: `http://localhost:17800/mcp`

## Host mount

`docker-compose.yml` mounts `D:\whisper-data` → `/data`. Drop video/audio files there (or any subdir). Cleanup happens automatically when you pass `cleanup: true`.

YouTube downloads land in `D:\whisper-data\youtube\` and are deleted by default after successful transcription.

## Wire up Claude Code

Add to `.mcp.json` in any project (or global `~/.claude.json` `mcpServers`):

```json
{
  "mcpServers": {
    "whisper": {
      "type": "http",
      "url": "http://localhost:17800/mcp"
    }
  }
}
```

Then in Claude Code: `/mcp` to confirm `whisper` is connected, and tools `transcribe_file` / `transcribe_youtube` are available.

## Tool examples

```json
// transcribe_file
{ "path": "lecture.mp4", "language": "en", "cleanup": false }
// or
{ "path": "D:\\whisper-data\\lecture.mp4", "language": "en" }

// transcribe_youtube
{ "url": "https://www.youtube.com/watch?v=...", "language": "ru" }
```

Progress notifications fire if the MCP client sends a `progressToken` (Claude Code does).

## CLI usage (no MCP client needed)

Plain HTTP endpoints sit alongside the MCP transport, so you can transcribe from any shell:

- `POST /api/transcribe/upload?filename=foo.mp4&language=ru` — raw body is the file bytes. Use for files that don't live under `D:\whisper-data`. Always cleaned up server-side.
- `POST /api/transcribe/file` — JSON `{"path": "...", "language": "ru", "cleanup": false}`. Path must resolve under `DATA_DIR`.
- `POST /api/transcribe/youtube` — JSON `{"url": "...", "language": "ru", "cleanup": true}`.

Each returns `{"text": "...", "language": "ru", "segments": 42, "elapsed_ms": 4732}`.

Two scripts wrap that for you:

### PowerShell (Windows Terminal / pwsh)

Local files are uploaded as bytes, so the file can live in `Downloads\`, `Desktop\`, or anywhere — not just `D:\whisper-data\`.

```powershell
# from any directory
.\whisper.ps1 .\notes.mp4
.\whisper.ps1 "$env:USERPROFILE\Downloads\interview.m4a" -Language ru
.\whisper.ps1 "https://www.youtube.com/watch?v=Uu3L5KVlWoo" -Language ru

# pipeline / redirect (use -NoSave to skip the interactive prompt)
.\whisper.ps1 .\notes.mp4 -NoSave > notes.txt
.\whisper.ps1 .\notes.mp4 -NoSave | Set-Content -Encoding utf8 notes.txt
```

Transcript goes to stdout; `[whisper] lang=… segments=… elapsed_ms=…` goes to stderr (so `> out.txt` captures just the text).

To call it as `whisper` from anywhere, add a function to your PowerShell profile:

```powershell
function whisper { & C:\dev\lab\rust-whisper-mcp\scripts\cli\whisper.ps1 @args }
```

### WSL / Linux bash

```bash
./scripts/cli/whisper.sh ./notes.mp4
./scripts/cli/whisper.sh /mnt/c/Users/$USER/Downloads/interview.m4a --language ru
./scripts/cli/whisper.sh "https://www.youtube.com/watch?v=Uu3L5KVlWoo" --language ru
```

Install `jq` for parsed output; otherwise raw JSON is printed.

## GPU selection

Docker Desktop on Windows ignores `device_ids:` in the compose `deploy.resources.reservations.devices` block and exposes all GPUs. The real selector is `CUDA_VISIBLE_DEVICES` in `environment:` — currently `"1"` (GPU2 / RTX 2080). Change to `"0"` or `"0,1"` to target a different card.

## Local dev (without Docker)

```bash
# needs: CUDA toolkit, clang, cmake, ffmpeg, yt-dlp on PATH
export WHISPER_MODEL_PATH=/path/to/ggml-large-v3.bin
export DATA_DIR=/tmp/whisper-data
cargo run --release
```

## Layout

```
src/
  main.rs       # axum: /health, /mcp (MCP), /api/transcribe/* (REST)
  handler.rs    # WhisperServer: shared run_file / run_youtube + MCP tool wrappers
  whisper.rs    # WhisperContext loading + transcribe_pcm (Mutex-serialized)
  media.rs      # ffmpeg decode, yt-dlp download, cleanup
  paths.rs      # path normalization (Windows / WSL / container forms → /data/*)
Dockerfile      # multi-stage cargo-chef + CUDA runtime
docker-compose.yml
scripts/
  entrypoint.sh
  cli/
    whisper.ps1 # PowerShell client (Windows Terminal / pwsh)
    whisper.sh  # bash client (WSL / Linux)
```

## Known limits

- Single GPU context, serialized — concurrent tool calls queue.
- Model load is ~5–10s on first call; held in memory for the container's lifetime.
- yt-dlp may need cookies for age-restricted/private videos — extend `media::download_youtube` with `--cookies-from-browser` or mount a cookies file if needed.
- Both the `/mcp` Streamable HTTP transport and the `/api/transcribe/*` REST endpoints are unauthenticated. Localhost only by default; do not expose port 17800 publicly without adding auth.
