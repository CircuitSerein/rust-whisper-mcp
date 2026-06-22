# Contributing

Thanks for your interest in `rust-whisper-mcp`. This is a small, focused project — a local MCP/REST server wrapping Whisper large-v3 on GPU. Contributions that keep it small and sharp are very welcome.

## Ground rules

- Keep PRs focused. One concern per PR.
- Match the surrounding style. Code is formatted with `rustfmt` and linted with `clippy`.
- Don't add a dependency without a clear reason — the dependency tree is intentionally lean.
- The `/mcp` and `/api/transcribe/*` endpoints are unauthenticated and localhost-only by design. Don't widen the network surface without a discussion in an issue first.

## Development setup

The production build links CUDA (the `cuda` feature is on by default). For day-to-day work you usually don't need a GPU:

```bash
# Format + lint (no GPU, no CUDA toolkit needed)
cargo fmt --all -- --check
cargo clippy --no-default-features --all-targets -- -D warnings

# Unit tests (path-resolution / traversal logic — pure, no GPU)
cargo test --no-default-features

# Compile check without CUDA (needs cmake + clang/libclang for whisper.cpp)
cargo build --no-default-features
```

`--no-default-features` drops the `cuda` feature so whisper.cpp builds for CPU. This is exactly what CI does. To build the real GPU binary, drop the flag (requires the CUDA toolkit, `nvcc`, cmake, and clang):

```bash
cargo build --release            # GPU build (cuda feature on)
```

Or just use Docker, which is the supported path:

```bash
cp .env.example .env             # edit WHISPER_DATA_DIR / GPU_INDEX / WHISPER_PORT
docker compose build
docker compose up -d
```

See the README for the full Docker / GPU pre-flight.

## Before you open a PR

- [ ] `cargo fmt --all -- --check` is clean
- [ ] `cargo clippy --no-default-features --all-targets -- -D warnings` is clean
- [ ] `cargo test --no-default-features` passes
- [ ] New behavior is covered by a test where practical
- [ ] No secrets, tokens, or machine-specific absolute paths committed

## Commit messages

Conventional-commit style is appreciated but not enforced:

```
feat: add language hint passthrough to the REST upload endpoint
fix: reject symlink escapes in resolve_in_data_dir
docs: clarify GPU selection on Docker Desktop
```

## Reporting bugs / requesting features

Open an issue using the templates. For transcription-quality problems, include the input characteristics (language, length, source — mic vs. file vs. YouTube) and, if possible, a short sample that reproduces it.
