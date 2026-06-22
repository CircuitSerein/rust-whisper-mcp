# Third-Party Licenses

`rust-whisper-mcp` is MIT-licensed (see [LICENSE](LICENSE)). It links and — in the
Docker image — bundles third-party components under their own licenses. This file
is an informational summary, not a substitute for each project's full license text.

For an exhaustive, machine-generated list of every Rust crate and its license, run:

```bash
cargo install cargo-about
cargo about generate about.hbs   # or: cargo about generate --format json
```

## Compiled into the binary

| Component | License | Notes |
| --- | --- | --- |
| [whisper.cpp / ggml](https://github.com/ggerganov/whisper.cpp) (Georgi Gerganov) | MIT | C/C++ inference engine, statically compiled in via `whisper-rs-sys`. |
| [whisper-rs](https://github.com/tazz4843/whisper-rs) | MIT / Unlicense | Rust bindings to whisper.cpp. |
| Rust crate dependencies (`rmcp`, `axum`, `tokio`, `serde`, `anyhow`, …) | mostly MIT / Apache-2.0 | See `Cargo.lock` / `cargo about` for the full set. |

### whisper.cpp / ggml (MIT)

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction ... (full text at the project repository).
```

## Bundled in the Docker image (redistributed only if you share the image)

| Component | License | Notes |
| --- | --- | --- |
| [ffmpeg](https://ffmpeg.org/legal.html) | LGPL-2.1+ / GPL (build-dependent) | Installed from Ubuntu apt. Audio decode. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense | Installed via pip. URL download. |
| [Deno](https://github.com/denoland/deno) | MIT | Installed for yt-dlp's JS challenge solver. |
| NVIDIA CUDA runtime | NVIDIA proprietary EULA | Base image `nvidia/cuda:*-runtime`. Redistribution governed by NVIDIA's terms. |
| `ggml-large-v3.bin` model weights | MIT | Downloaded at runtime from HuggingFace `ggerganov/whisper.cpp`. |

If you publish or redistribute the built Docker image, review the ffmpeg build
flavor and the NVIDIA CUDA redistribution terms — those carry obligations beyond
this project's MIT license.
