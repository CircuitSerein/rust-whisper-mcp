use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use tokio::process::Command;

/// Decode any media file → 16kHz mono f32 PCM via ffmpeg.
///
/// We emit raw little-endian s16 instead of WAV because ffmpeg cannot seek
/// back on a pipe to patch the RIFF chunk size, which produces a truncated
/// header that hound rejects.
pub async fn decode_to_pcm(input: &Path) -> Result<Vec<f32>> {
    let started = Instant::now();
    let out = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
        ])
        .arg(input)
        .args([
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn ffmpeg")?;

    if !out.status.success() {
        return Err(anyhow!(
            "ffmpeg failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let bytes = out.stdout;
    if bytes.len() % 2 != 0 {
        return Err(anyhow!("ffmpeg s16 output has odd byte length: {}", bytes.len()));
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();

    let audio_secs = samples.len() as f64 / 16_000.0;
    tracing::info!(
        input = %input.display(),
        samples = samples.len(),
        audio_secs = format!("{audio_secs:.2}"),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "ffmpeg decode complete"
    );
    Ok(samples)
}

/// Download a YouTube (or any yt-dlp-supported) URL as the smallest available
/// audio-only stream, returning the resulting file path inside `dest_dir`.
///
/// We skip the `--audio-format wav` postprocess step — `decode_to_pcm` runs
/// ffmpeg anyway, so an extra wav conversion is wasted work. The smallest
/// audio stream is enough since we downsample to 16 kHz mono regardless.
pub async fn download_youtube(url: &str, dest_dir: &Path) -> Result<PathBuf> {
    let started = Instant::now();
    tokio::fs::create_dir_all(dest_dir).await.ok();
    let id = uuid::Uuid::new_v4();
    let out_template = dest_dir.join(format!("{id}.%(ext)s"));
    tracing::info!(url, "yt-dlp download starting");

    let status = Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "--no-progress",
            "--quiet",
            // YouTube now uses JS-based "n" challenge tokens. We have deno
            // installed; this flag lets yt-dlp also fetch the EJS solver
            // script from GitHub (cached after first run).
            "--remote-components", "ejs:github",
            // Modern format selection: best audio-only stream, sorted to prefer
            // smallest size / lowest bitrate first. Falls back to "ba" alone if
            // a source only exposes muxed video+audio.
            "-f", "ba/bestaudio",
            "-S", "+size,+br",
            "-o",
        ])
        .arg(&out_template)
        .arg(url)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("spawn yt-dlp")?;

    if !status.success() {
        return Err(anyhow!("yt-dlp failed: {status}"));
    }

    // Find the produced file (extension depends on source — m4a/webm/opus/etc.)
    let prefix = format!("{id}.");
    let mut entries = tokio::fs::read_dir(dest_dir).await.context("read dest_dir")?;
    while let Some(entry) = entries.next_entry().await.context("read_dir entry")? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix) {
            let path = entry.path();
            let size = tokio::fs::metadata(&path).await.map(|m| m.len()).unwrap_or(0);
            tracing::info!(
                path = %path.display(),
                bytes = size,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "yt-dlp download complete"
            );
            return Ok(path);
        }
    }
    Err(anyhow!("yt-dlp produced no file matching {prefix}*"))
}

pub async fn delete_quiet(p: &Path) {
    if let Err(e) = tokio::fs::remove_file(p).await {
        tracing::warn!(path = %p.display(), error = %e, "cleanup failed");
    }
}
