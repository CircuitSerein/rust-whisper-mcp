# rust-whisper-mcp

[![CI](https://github.com/CircuitSerein/rust-whisper-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/CircuitSerein/rust-whisper-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Local MCP server: Whisper large-v3 on a dedicated NVIDIA GPU via Docker. Transcribes local media or any yt-dlp URL — and **analyzes video**, returning keyframes as images alongside a timestamped transcript so a vision model (Claude in Claude Code, etc.) can "see" the video. Think of it as a simple, self-hosted take on Google Gemini–style video understanding, on your own GPU. Written in Rust (rmcp + whisper-rs CUDA + ffmpeg 8.1).

Tools: `transcribe_file`, `transcribe_youtube`, `analyze_file`, `analyze_youtube`, `extract_frames`, `estimate_video_cost`.

## Stack

- Rust + [`rmcp`](https://docs.rs/rmcp) 2.2 — MCP server, Streamable HTTP transport
- [`whisper-rs`](https://docs.rs/whisper-rs) 0.16 with `cuda` feature — GGML large-v3 on GPU
- `ffmpeg` (any → 16kHz mono PCM) + `yt-dlp` (YouTube → wav), on a uv-managed Python 3.13
- [`bgutil-ytdlp-pot-provider`](https://github.com/Brainicism/bgutil-ytdlp-pot-provider) — sidecar that mints the PO tokens YouTube now requires
- Docker Compose, CUDA 12.4 runtime, single GPU selected by `GPU_INDEX` (default `1`, 0-indexed)

## Pre-flight (run BEFORE first build)

In PowerShell on the host, verify your target GPU is reachable from Docker (index `1` by default, 0-indexed — matches `GPU_INDEX`):

```powershell
docker run --rm --gpus '"device=1"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You must see that GPU listed. If not — install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and enable GPU support in Docker Desktop → Settings → Resources → WSL Integration / GPU.

## Configure

```bash
git clone https://github.com/CircuitSerein/rust-whisper-mcp.git
cd rust-whisper-mcp
cp .env.example .env   # then edit
```

`.env` controls the machine-specific bits (compose reads it):

| Variable           | Default          | Meaning                                                    |
| ------------------ | ---------------- | ---------------------------------------------------------- |
| `WHISPER_DATA_DIR` | `D:/whisper-data`| Host dir mounted at `/data`. Drop media here.              |
| `GPU_INDEX`        | `1`              | Physical GPU to expose (0-indexed).                        |
| `WHISPER_PORT`     | `17800`          | Host port (container always listens on 8080).              |
| `RUST_LOG`         | `info`           | Log level.                                                 |

On Windows / Docker Desktop, use forward slashes in `WHISPER_DATA_DIR` (e.g. `D:/whisper-data`).

## Build & run

```powershell
# from the repo root, after `cp .env.example .env`
docker compose build
docker compose up -d
docker compose logs -f whisper-mcp
```

First start downloads `ggml-large-v3.bin` (~3 GB) into the `whisper-models` volume. Subsequent starts skip that.

Health: `curl http://localhost:17800/health` → `ok`
MCP endpoint: `http://localhost:17800/mcp`

## Host mount

`docker-compose.yml` mounts `WHISPER_DATA_DIR` (default `D:\whisper-data`) → `/data`. Drop video/audio files there (or any subdir). Cleanup happens automatically when you pass `cleanup: true`.

YouTube downloads land in `<WHISPER_DATA_DIR>\youtube\` and are deleted by default after successful transcription.

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
// timed [MM:SS] lines (to anchor frame windows) instead of plain text
{ "path": "lecture.mp4", "timestamps": true }

// transcribe_youtube
{ "url": "https://www.youtube.com/watch?v=...", "language": "ru" }
```

`transcribe_*` is the **light default**: it returns the transcript plus a one-line info
header (`Video 1920×1080 · 04:58 · lang en`) and **no images** — reach for the video
tools below only when you actually need to *see* frames.

Progress notifications fire if the MCP client sends a `progressToken` (Claude Code does).

## Video analysis

When you need the model to *see* the video, `analyze_file` / `analyze_youtube` extract
keyframes and transcribe in one call, returning a mixed result: metadata, then the
`[MM:SS] text` transcript, then each frame as an image labelled `[frame N] MM:SS` — so
the model can line up what's on screen with what's said. Best for a **short clip**; for a
long video, transcribe once and page `extract_frames` (below). `estimate_video_cost`
previews the token cost first (no whisper, no images); `extract_frames` returns frames
only (no transcript). Near-identical consecutive frames are dropped automatically
(`dedup`, default on) so a static stretch never returns a dozen copies of one shot.

```json
// estimate first (cheap — no images, no transcription)
{ "path": "recipe.mp4", "quality": "medium", "sampling": "scene" }
// then analyze
{ "path": "recipe.mp4", "quality": "medium", "sampling": "scene", "max_frames": 10 }
// youtube, uniform sampling, grid contact-sheet layout
{ "url": "https://youtu.be/...", "sampling": "uniform", "fps": 0.5, "layout": "grid" }
```

Options: `quality` = `compact` (640px) · `low` (854) · `medium` (1280, default) · `high`
(1568 — the recommended ingest ceiling; larger is pointless/rejected). `sampling` =
`scene` (default, on scene changes; tune `scene_threshold` 0.1–0.6) or `uniform` (`fps`).
`dedup` (default `true`) drops near-identical consecutive frames via ffmpeg `mpdecimate`,
so `max_frames` is an upper bound — set `false` for a strict fixed-rate filmstrip.
`max_frames` defaults to 10, hard-capped at 60. `layout` = `frames` (default) or `grid`.
GPU (NVDEC) decode is used when available (`FFMPEG_HWACCEL`, default `auto`).

The transcript is emitted **before** the frames on purpose: one MCP tool result is
capped (`MAX_MCP_OUTPUT_TOKENS`, ~25k default ≈ ~18 medium frames), and the client
truncates the tail — so the cheap, always-useful transcript survives while the
expensive frames are what get cut if anything does.

### Whole videos of any length

A single call can't hold a long video (that token cap). To analyze the whole thing,
**transcribe once, then page keyframes by time window** — `extract_frames` takes
`start_secs`/`end_secs`, and frame timestamps stay absolute so they still line up
with the transcript:

```json
// full [MM:SS] transcript once (cheap, no images):
// transcribe_file → { "path": "talk.mp4", "timestamps": true }
// then walk the video window by window (frames only; dedup drops static dupes)
{ "path": "talk.mp4", "start_secs": 0,   "end_secs": 120, "quality": "medium", "max_frames": 12 }
{ "path": "talk.mp4", "start_secs": 120, "end_secs": 240, "quality": "medium", "max_frames": 12 }
// ... until end_secs reaches the duration
```

Keep each window lean — frames are heavy (~1.2k tokens each at medium) and accumulate;
prefer `scene` sampling and reserve `medium` for the few windows where you must read
on-screen text. Cranking `max_frames` high is a false economy even with `MAX_MCP_OUTPUT_TOKENS`
raised — the binding limit becomes the reader's own context.

For hands-off use there's a **`video-analyst` subagent** (`.claude/agents/video-analyst.md`):
it does this loop in its own isolated context — transcribe once, page the frames,
keep running notes — and returns a compact text report, so hundreds of frames never
pollute the main conversation. Ask Claude Code to "use the video-analyst to summarize
`<file or url>`". The read-only whisper tools are allow-listed in
`.claude/settings.json` so its paginated loop runs unattended. (`analyze_file` /
`analyze_youtube` remain the direct, single-call path for short clips when you want to
see the frames yourself.)

## CLI usage (no MCP client needed)

Plain HTTP endpoints sit alongside the MCP transport, so you can transcribe from any shell:

- `POST /api/transcribe/upload?filename=foo.mp4&language=ru` — raw body is the file bytes. Use for files that don't live under `D:\whisper-data`. Always cleaned up server-side.
- `POST /api/transcribe/file` — JSON `{"path": "...", "language": "ru", "cleanup": false}`. Path must resolve under `DATA_DIR`.
- `POST /api/transcribe/youtube` — JSON `{"url": "...", "language": "ru", "cleanup": true}`.
- `POST /api/estimate` — JSON `{"path": "..."}` or `{"url": "..."}` (+ `quality`, `sampling`, `max_frames`). Returns the token estimate, no images.
- `POST /api/analyze/file` — JSON `{"path": "...", "quality": "medium", "max_frames": 10, "layout": "frames"}`. Returns metadata, base64 frames, and the timestamped transcript.
- `POST /api/analyze/youtube` — JSON `{"url": "...", ...}`. Same shape as analyze/file.
- `POST /api/extract-frames` — JSON `{"path": "...", "start_secs": 0, "end_secs": 120, ...}`. Frames only, no transcript; `start_secs`/`end_secs` page a long video by time window (absolute `[MM:SS]`).

Each returns `{"text": "...", "language": "ru", "segments": 42, "elapsed_ms": 4732}`.

Two scripts wrap that for you:

### PowerShell (Windows Terminal / pwsh)

Local files are uploaded as bytes, so the file can live in `Downloads\`, `Desktop\`, or anywhere — not just `D:\whisper-data\`.

```powershell
# from the repo root (or define the `whisper` function below to call from anywhere)
.\scripts\cli\whisper.ps1 .\notes.mp4
.\scripts\cli\whisper.ps1 "$env:USERPROFILE\Downloads\interview.m4a" -Language ru
.\scripts\cli\whisper.ps1 "https://www.youtube.com/watch?v=Uu3L5KVlWoo" -Language ru

# pipeline / redirect (use -NoSave to skip the interactive prompt)
.\scripts\cli\whisper.ps1 .\notes.mp4 -NoSave > notes.txt
.\scripts\cli\whisper.ps1 .\notes.mp4 -NoSave | Set-Content -Encoding utf8 notes.txt
```

Transcript goes to stdout; `[whisper] lang=… segments=… elapsed_ms=…` goes to stderr (so `> out.txt` captures just the text).

To call it as `whisper` from anywhere, add a function to your PowerShell profile:

```powershell
function whisper { & C:\path\to\rust-whisper-mcp\scripts\cli\whisper.ps1 @args }
```

### WSL / Linux bash

```bash
./scripts/cli/whisper.sh ./notes.mp4
./scripts/cli/whisper.sh /mnt/c/Users/$USER/Downloads/interview.m4a --language ru
./scripts/cli/whisper.sh "https://www.youtube.com/watch?v=Uu3L5KVlWoo" --language ru
```

Install `jq` for parsed output; otherwise raw JSON is printed.

## GPU selection

Docker Desktop on Windows ignores `device_ids:` in the compose `deploy.resources.reservations.devices` block and exposes all GPUs. The real selector is `CUDA_VISIBLE_DEVICES`, which compose sets from `GPU_INDEX` in `.env` (default `1`). Set `GPU_INDEX=0` to target the first card. To expose several cards, set `CUDA_VISIBLE_DEVICES` directly in `docker-compose.yml` (e.g. `"0,1"`).

## Local dev (without Docker)

```bash
# needs: CUDA toolkit, clang, cmake, ffmpeg, yt-dlp on PATH
#   yt-dlp with the PO-token plugin:
#   uv tool install yt-dlp --with bgutil-ytdlp-pot-provider
#   plus the provider itself:
#   docker run -d -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
export WHISPER_MODEL_PATH=/path/to/ggml-large-v3.bin
export DATA_DIR=/tmp/whisper-data
export YTDLP_POT_BASE_URL=http://127.0.0.1:4416
cargo run --release            # GPU build (cuda feature on by default)
```

The `cuda` feature is on by default. For CPU-only work (lint, tests, a compile
check without the CUDA toolkit — only `clang` + `cmake` needed) disable it:

```bash
cargo clippy --no-default-features --all-targets -- -D warnings
cargo test  --no-default-features          # path-resolution unit tests
cargo build --no-default-features          # CPU whisper.cpp build
```

This is exactly what CI runs. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev loop.

## Layout

```
src/
  main.rs       # axum: /health, /mcp (MCP), /api/transcribe/* + /api/analyze/* (REST)
  handler.rs    # WhisperServer: shared run_* cores (transcribe + analyze) + MCP tool wrappers
  whisper.rs    # WhisperContext loading + transcribe_pcm (Mutex-serialized), timed segments
  media.rs      # ffmpeg decode + keyframe extraction, yt-dlp audio/video download, ffprobe
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
- YouTube downloads need the `bgutil-provider` sidecar (compose starts it). YouTube serves media URLs only to clients presenting a PO token; without the provider every download fails with `HTTP Error 403: Forbidden`. `YTDLP_PLAYER_CLIENTS` (default `tv_simply`) picks the client whose formats stay plain-https — set it to `default` to hand the choice back to yt-dlp if YouTube shifts again.
- Age-restricted / members-only videos need cookies: export a Netscape `cookies.txt` into `WHISPER_DATA_DIR` and set `YTDLP_COOKIES=/data/cookies.txt`.
- Audio is picked with `-S lang,…`, so the ORIGINAL track wins over YouTube's auto-dubs. Without that, a Russian video can come back transcribed from its English dub.
- Both the `/mcp` Streamable HTTP transport and the `/api/transcribe/*` REST endpoints are **unauthenticated**. The container is published on `127.0.0.1` only (see `docker-compose.yml`), so it is not reachable from the LAN by default — don't switch the host bind to `0.0.0.0` without putting auth / a reverse proxy in front.
- `transcribe_file`'s path sandbox is **lexical**: it rejects `..` and keeps paths under `DATA_DIR`, but does not resolve symlinks. Treat `DATA_DIR` as a trusted directory.
- `transcribe_youtube` runs `yt-dlp`, which fetches and executes a remote EJS "n-challenge" solver from GitHub (`--remote-components ejs:github`) via deno. Drop that flag in `media.rs` for a fully offline / locked-down deployment. Only `http`/`https` URLs are accepted.
- `/api/transcribe/upload` buffers the whole body in memory (4 GiB cap), so a large upload transiently uses ~2× its size in RAM.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev loop, lint/test commands, and scope notes.

## License

[MIT](LICENSE) © CircuitSerein

This project links and bundles third-party components under their own licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
