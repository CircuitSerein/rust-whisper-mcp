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
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
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
        return Err(anyhow!(
            "ffmpeg s16 output has odd byte length: {}",
            bytes.len()
        ));
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

/// Which YouTube player clients yt-dlp asks for formats, unless
/// `YTDLP_PLAYER_CLIENTS` overrides it.
///
/// YouTube hands out playable media URLs only to clients that present a PO
/// token. The clients yt-dlp reaches without one (`android_vr`) now answer the
/// media URL with HTTP 403, and the web clients are SABR-only, so their formats
/// arrive with no URL at all. `tv_simply` is the client whose formats stay
/// plain-https once a PO token is available — see [`ytdlp_extractor_args`].
const DEFAULT_PLAYER_CLIENTS: &str = "tv_simply";

/// Format-sort string shared by both download paths. `lang` comes first so the
/// ORIGINAL audio track wins: YouTube attaches dubbed tracks (fr-FR, pt-BR, …)
/// to the same video, and sorting by size alone happily picks a dub, which then
/// gets transcribed as if it were the source language.
const AUDIO_SORT: &str = "lang,+size,+br";

/// Read an env var, treating unset and blank as the same "not configured".
fn env_opt(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Build the `--extractor-args` / `--cookies` flags every yt-dlp call shares.
///
/// * `clients` — `YTDLP_PLAYER_CLIENTS`; `default` (or empty) leaves yt-dlp's
///   own client list alone.
/// * `pot_base_url` — `YTDLP_POT_BASE_URL`, the bgutil PO-token provider
///   (`brainicism/bgutil-ytdlp-pot-provider`). Without it YouTube downloads fail
///   with `HTTP Error 403: Forbidden`. Unset leaves the plugin's own default
///   (`http://127.0.0.1:4416`) in place.
/// * `cookies` — `YTDLP_COOKIES`, a Netscape cookies file for age-restricted or
///   members-only videos.
fn build_extractor_args(
    clients: Option<&str>,
    pot_base_url: Option<&str>,
    cookies: Option<&str>,
) -> Vec<String> {
    let mut args = Vec::new();
    let clients = clients.unwrap_or(DEFAULT_PLAYER_CLIENTS);
    if !clients.is_empty() && clients != "default" {
        args.push("--extractor-args".to_string());
        args.push(format!("youtube:player_client={clients}"));
    }
    if let Some(base) = pot_base_url {
        args.push("--extractor-args".to_string());
        args.push(format!("youtubepot-bgutilhttp:base_url={base}"));
    }
    if let Some(path) = cookies {
        args.push("--cookies".to_string());
        args.push(path.to_string());
    }
    args
}

/// Env-backed wrapper over [`build_extractor_args`].
fn ytdlp_extractor_args() -> Vec<String> {
    build_extractor_args(
        env_opt("YTDLP_PLAYER_CLIENTS").as_deref(),
        env_opt("YTDLP_POT_BASE_URL").as_deref(),
        env_opt("YTDLP_COOKIES").as_deref(),
    )
}

/// Trim a yt-dlp stderr dump down to something that fits in an MCP error.
fn stderr_tail(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let tail: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .rev()
        .take(3)
        .collect();
    tail.into_iter().rev().collect::<Vec<_>>().join("; ")
}

/// Download a YouTube (or any yt-dlp-supported) URL as the smallest available
/// audio-only stream, returning the resulting file path inside `dest_dir`.
///
/// We skip the `--audio-format wav` postprocess step — `decode_to_pcm` runs
/// ffmpeg anyway, so an extra wav conversion is wasted work. The smallest
/// audio stream is enough since we downsample to 16 kHz mono regardless.
pub async fn download_youtube(url: &str, dest_dir: &Path) -> Result<PathBuf> {
    // Validate the scheme before handing the URL to yt-dlp. yt-dlp will happily
    // accept file:// and other local schemes, which would turn this into an
    // arbitrary host-file reader; restrict to http(s).
    let parsed = url::Url::parse(url).with_context(|| format!("invalid URL: {url}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(anyhow!(
            "unsupported URL scheme '{}' — only http/https are allowed",
            parsed.scheme()
        ));
    }

    let started = Instant::now();
    tokio::fs::create_dir_all(dest_dir).await.ok();
    let id = uuid::Uuid::new_v4();
    let out_template = dest_dir.join(format!("{id}.%(ext)s"));
    tracing::info!(url, "yt-dlp download starting");

    let out = Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "--no-progress",
            "--quiet",
            // YouTube now uses JS-based "n" challenge tokens. We have deno
            // installed; this flag lets yt-dlp also fetch the EJS solver
            // script from GitHub (cached after first run).
            "--remote-components",
            "ejs:github",
            // Modern format selection: best audio-only stream, original language
            // first, then smallest size / lowest bitrate. Falls back to "ba"
            // alone if a source only exposes muxed video+audio.
            "-f",
            "ba/bestaudio",
            "-S",
            AUDIO_SORT,
        ])
        .args(ytdlp_extractor_args())
        .arg("-o")
        .arg(&out_template)
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn yt-dlp")?;

    if !out.status.success() {
        cleanup_prefix(dest_dir, &format!("{id}.")).await;
        return Err(anyhow!(
            "yt-dlp failed: {}: {}",
            out.status,
            stderr_tail(&out.stderr)
        ));
    }

    // Find the produced file (extension depends on source — m4a/webm/opus/etc.)
    let prefix = format!("{id}.");
    let mut entries = tokio::fs::read_dir(dest_dir)
        .await
        .context("read dest_dir")?;
    while let Some(entry) = entries.next_entry().await.context("read_dir entry")? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(&prefix) {
            let path = entry.path();
            let size = tokio::fs::metadata(&path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
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

/// Recursively remove a directory, warning (not failing) on error. Used to clean
/// the temporary keyframe workdir on every exit path.
pub async fn delete_dir_quiet(p: &Path) {
    if let Err(e) = tokio::fs::remove_dir_all(p).await {
        tracing::warn!(path = %p.display(), error = %e, "temp dir cleanup failed");
    }
}

// ============================= video analysis =============================

/// Video metadata from ffprobe.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    /// Video dimensions, or `0` for an audio-only input (no video stream).
    pub width: u32,
    pub height: u32,
    /// Duration in seconds; `NAN` when ffprobe cannot determine it (some webm/
    /// streamed outputs) — callers fall back to the decoded PCM length.
    pub duration_secs: f64,
    pub has_audio: bool,
}

/// One extracted keyframe.
#[derive(Debug, Clone)]
pub struct Frame {
    /// 1-based position in the returned sequence.
    pub index: usize,
    /// Source timestamp in seconds (from the metadata sidecar).
    pub pts_time: f64,
    pub jpeg: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    /// Emit a frame on each scene change above `threshold` (plus frame 0).
    Scene { threshold: f64 },
    /// Emit `fps` frames per second (may be < 1).
    Uniform { fps: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct ExtractParams {
    /// Long-edge cap in px; `None` = source resolution.
    pub box_px: Option<u32>,
    pub sampling: SamplingMode,
    pub max_frames: u32,
    /// Drop near-identical consecutive frames via ffmpeg `mpdecimate` before
    /// scaling/printing. Collapses static runs (e.g. a held end-card or a paused
    /// screencast) so a uniform sample doesn't return N copies of one shot.
    /// Near-no-op for scene sampling, which already picks distinct frames.
    pub dedup: bool,
    /// Extract only from this source time onward (seconds). `None`/≤0 = from start.
    /// Frame timestamps are reported ABSOLUTE (offset by this), so `[MM:SS]` stays
    /// correct when paginating a long video one window at a time.
    pub start_secs: Option<f64>,
    /// Extract only up to this source time (seconds). `None` = to end of video.
    pub end_secs: Option<f64>,
}

/// Probe a media file for video dimensions, duration, and audio-stream presence.
pub async fn probe_video(input: &Path) -> Result<VideoMeta> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=width,height,codec_type:stream_disposition=attached_pic:format=duration",
            "-of",
            "json",
        ])
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn ffprobe")?;
    if !out.status.success() {
        return Err(anyhow!(
            "ffprobe failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse_probe_json(&out.stdout)
}

/// Extract keyframes to raw JPEG bytes. Writes to a uuid temp dir under
/// `work_root` (never any user-supplied path segment) and cleans it up on every
/// exit path. Returns frames in order with their source timestamps; an empty Vec
/// means no frames matched (e.g. a static screencast with no scene cuts) — the
/// caller decides whether to fall back to uniform sampling.
pub async fn extract_keyframes(
    input: &Path,
    work_root: &Path,
    params: &ExtractParams,
) -> Result<Vec<Frame>> {
    tokio::fs::create_dir_all(work_root).await.ok();
    let dir = work_root.join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&dir)
        .await
        .context("create frame temp dir")?;
    let result = extract_into(input, &dir, params).await;
    delete_dir_quiet(&dir).await;
    result
}

async fn extract_into(input: &Path, dir: &Path, params: &ExtractParams) -> Result<Vec<Frame>> {
    let pts_file = dir.join("pts.txt");
    let out_pattern = dir.join("f_%05d.jpg");
    let vf = build_filtergraph(params, &pts_file);

    // Optional time window: input-seek `-ss` (fast) from `start`, `-t` for its
    // duration. No `-copyts`, so ffmpeg re-bases timestamps to 0 at the seek point;
    // we add `window_start` back below to report ABSOLUTE times.
    let window_start = params.start_secs.filter(|s| s.is_finite() && *s > 0.0);
    let from = window_start.unwrap_or(0.0);
    let dur = params
        .end_secs
        .filter(|e| e.is_finite() && *e > from)
        .map(|e| e - from);

    // GPU decode first (NVDEC — separate silicon from whisper's CUDA cores), then
    // CPU fallback so a missing/unsupported GPU or odd codec never breaks
    // extraction. `-hwaccel auto` self-falls-back within the process; the explicit
    // second attempt covers a forced `FFMPEG_HWACCEL` that errors out.
    let mode = hwaccel_mode();
    let attempts: Vec<Option<String>> = if mode == "none" {
        vec![None]
    } else {
        vec![Some(mode), None]
    };
    let mut last_err = None;
    for attempt in &attempts {
        clear_dir_files(dir).await; // fresh slate per attempt
        match run_ffmpeg_extract(
            input,
            &vf,
            &out_pattern,
            attempt.as_deref(),
            window_start,
            dur,
        )
        .await
        {
            Ok(()) => {
                last_err = None;
                break;
            }
            Err(e) => {
                tracing::warn!(hwaccel = ?attempt, error = %e, "ffmpeg extract attempt failed");
                last_err = Some(e);
            }
        }
    }
    if let Some(e) = last_err {
        return Err(e);
    }

    // Timestamps from the metadata sidecar (log-level independent, 1:1 ordered).
    // Add `from` so a windowed extract reports ABSOLUTE source times, not window-
    // relative ones (ffmpeg re-based them to 0 at the seek point).
    let pts: Vec<f64> = parse_metadata_pts(
        &tokio::fs::read_to_string(&pts_file)
            .await
            .unwrap_or_default(),
    )
    .into_iter()
    .map(|t| t + from)
    .collect();

    // Frame files, sorted (zero-padded name ⇒ lexical order == numeric order).
    let mut files: Vec<PathBuf> = Vec::new();
    let mut rd = tokio::fs::read_dir(dir).await.context("read frame dir")?;
    while let Some(e) = rd.next_entry().await.context("frame dir entry")? {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) == Some("jpg") {
            files.push(p);
        }
    }
    files.sort();

    if files.is_empty() {
        return Ok(Vec::new());
    }
    if files.len() != pts.len() {
        // Both are emission-ordered, so the leading min(files,pts) still align
        // correctly; we drop any unlabelled tail rather than emit a wrong [MM:SS].
        tracing::warn!(
            frames = files.len(),
            pts = pts.len(),
            "frame/pts count mismatch — using aligned prefix"
        );
    }

    let n = files.len().min(pts.len());
    let mut frames: Vec<Frame> = Vec::with_capacity(n);
    for (path, &t) in files.iter().take(n).zip(pts.iter().take(n)) {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("read frame {}", path.display()))?;
        if bytes.is_empty() {
            tracing::warn!(path = %path.display(), "empty frame file, skipping");
            continue;
        }
        frames.push(Frame {
            index: frames.len() + 1,
            pts_time: t,
            jpeg: bytes,
        });
    }

    // Cap to max_frames, evenly spread, then re-index 1..=len contiguously.
    let mut frames = subsample_even(&frames, params.max_frames as usize);
    for (i, f) in frames.iter_mut().enumerate() {
        f.index = i + 1;
    }
    Ok(frames)
}

fn build_filtergraph(params: &ExtractParams, pts_file: &Path) -> String {
    // Single-quote the sub-expressions so their inner commas are literal (the arg
    // is passed to ffmpeg without a shell). `eq(n,0)+` guarantees the t=0 frame.
    let select = match params.sampling {
        SamplingMode::Scene { threshold } => format!("select='eq(n,0)+gt(scene,{threshold})',"),
        SamplingMode::Uniform { fps } => format!("fps={fps},"),
    };
    // `mpdecimate` drops frames too similar to the last KEPT one. It removes them
    // from the stream entirely (downstream `scale`/`metadata` never see them), and
    // `-fps_mode vfr` (set in run_ffmpeg_extract) stops the muxer re-inserting them —
    // so the sidecar and the written JPEGs stay 1:1. It MUST sit before `metadata=print`
    // for that reason. After `select` (scene) it's a near-no-op; after `fps` (uniform)
    // it collapses static runs so a held shot yields one frame, not a dozen dupes.
    let dedup = if params.dedup { "mpdecimate," } else { "" };
    let scale = params
        .box_px
        .map(|b| format!("scale='min({b},iw)':'min({b},ih)':force_original_aspect_ratio=decrease,"))
        .unwrap_or_default();
    // `metadata=print` only emits its `frame:N … pts_time:T` header for frames that
    // carry at least one metadata entry. `select` (scene) attaches `lavfi.scene_score`
    // so its frames print; `fps` (uniform) attaches nothing, so the sidecar comes out
    // EMPTY and every timestamp is lost. Stamp a throwaway key first so the header
    // fires for every kept frame in both modes.
    format!(
        "{select}{dedup}{scale}metadata=add:key=ts:value=1,metadata=print:file={}",
        pts_file.to_string_lossy()
    )
}

async fn run_ffmpeg_extract(
    input: &Path,
    vf: &str,
    out_pattern: &Path,
    hwaccel: Option<&str>,
    start_secs: Option<f64>,
    dur_secs: Option<f64>,
) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error"]);
    // Input-seek `-ss` before `-i` (fast, demuxer-level). Both it and `-hwaccel`
    // are input options and must precede `-i`.
    if let Some(s) = start_secs {
        cmd.args(["-ss".to_string(), format!("{s}")]);
    }
    if let Some(hw) = hwaccel {
        // -hwaccel must precede -i to accelerate decoding of the input.
        cmd.args(["-hwaccel", hw]);
    }
    cmd.arg("-i").arg(input);
    // `-t` (duration) is an output option — it caps how much of the window is read.
    if let Some(d) = dur_secs {
        cmd.args(["-t".to_string(), format!("{d}")]);
    }
    cmd.args(["-an", "-vf", vf, "-fps_mode", "vfr", "-q:v", "2"])
        .arg(out_pattern)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = cmd.output().await.context("spawn ffmpeg (extract)")?;
    if !out.status.success() {
        return Err(anyhow!(
            "ffmpeg extract failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Download a URL keeping BOTH video and audio (capped ≤1080p, ≤2G), merged to a
/// single mp4, for the analyze path. `download_youtube` (audio-only) is unchanged
/// and still used by `transcribe_youtube`.
pub async fn download_video(url: &str, dest_dir: &Path) -> Result<PathBuf> {
    let parsed = url::Url::parse(url).with_context(|| format!("invalid URL: {url}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(anyhow!(
            "unsupported URL scheme '{}' — only http/https are allowed",
            parsed.scheme()
        ));
    }

    let started = Instant::now();
    tokio::fs::create_dir_all(dest_dir).await.ok();
    let id = uuid::Uuid::new_v4();
    let out_template = dest_dir.join(format!("{id}.%(ext)s"));
    tracing::info!(url, "yt-dlp video download starting");

    let out = Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "--no-progress",
            "--quiet",
            "--remote-components",
            "ejs:github",
            // Keep video+audio, cap height so the download stays sane; merge to mp4.
            "-f",
            "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
            // Original audio track first (see AUDIO_SORT) — a dubbed track would
            // desync the transcript from what the frames show.
            "-S",
            "lang",
            "--merge-output-format",
            "mp4",
            "--max-filesize",
            "2G",
            // Print the FINAL merged path so we never grab a pre-merge fragment.
            "--print",
            "after_move:filepath",
        ])
        .args(ytdlp_extractor_args())
        .arg("-o")
        .arg(&out_template)
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn yt-dlp")?;

    if !out.status.success() {
        cleanup_prefix(dest_dir, &format!("{id}.")).await;
        return Err(anyhow!(
            "yt-dlp failed: {}: {}",
            out.status,
            stderr_tail(&out.stderr)
        ));
    }

    // Primary: the exact path yt-dlp printed. Fallback: scan for the merged file.
    let printed = String::from_utf8_lossy(&out.stdout);
    if let Some(line) = printed.lines().map(str::trim).rev().find(|l| !l.is_empty()) {
        let p = PathBuf::from(line);
        if tokio::fs::metadata(&p).await.is_ok() {
            tracing::info!(
                path = %p.display(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "yt-dlp video download complete"
            );
            return Ok(p);
        }
    }
    find_merged_output(dest_dir, &id).await
}

/// Fallback discovery: pick the final container `{id}.<ext>` (exactly one dot in
/// the name — uuids contain no dots), skipping pre-merge fragments like
/// `{id}.f399.mp4` and `.part` files.
async fn find_merged_output(dir: &Path, id: &uuid::Uuid) -> Result<PathBuf> {
    let prefix = format!("{id}.");
    let mut rd = tokio::fs::read_dir(dir).await.context("read dest_dir")?;
    while let Some(e) = rd.next_entry().await.context("read_dir entry")? {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&prefix) || name.ends_with(".part") {
            continue;
        }
        let ext = e
            .path()
            .extension()
            .and_then(|x| x.to_str())
            .map(|s| s.to_ascii_lowercase());
        let is_container = matches!(ext.as_deref(), Some("mp4" | "mkv" | "webm" | "mov" | "m4v"));
        if is_container && name.matches('.').count() == 1 {
            return Ok(e.path());
        }
    }
    Err(anyhow!("yt-dlp produced no merged video for {prefix}*"))
}

async fn cleanup_prefix(dir: &Path, prefix: &str) {
    if let Ok(mut rd) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(e)) = rd.next_entry().await {
            if e.file_name().to_string_lossy().starts_with(prefix) {
                let _ = tokio::fs::remove_file(e.path()).await;
            }
        }
    }
}

async fn clear_dir_files(dir: &Path) {
    if let Ok(mut rd) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(e)) = rd.next_entry().await {
            let _ = tokio::fs::remove_file(e.path()).await;
        }
    }
}

/// `FFMPEG_HWACCEL`: `auto` (default) uses GPU decode when available and falls
/// back to software; `none` forces CPU; any other value is passed to `-hwaccel`.
fn hwaccel_mode() -> String {
    std::env::var("FFMPEG_HWACCEL")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "auto".to_string())
}

fn parse_probe_json(bytes: &[u8]) -> Result<VideoMeta> {
    #[derive(serde::Deserialize, Default)]
    struct Disposition {
        #[serde(default)]
        attached_pic: i64,
    }
    #[derive(serde::Deserialize)]
    struct Stream {
        width: Option<u32>,
        height: Option<u32>,
        codec_type: Option<String>,
        #[serde(default)]
        disposition: Disposition,
    }
    #[derive(serde::Deserialize, Default)]
    struct Format {
        duration: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct Probe {
        #[serde(default)]
        streams: Vec<Stream>,
        #[serde(default)]
        format: Format,
    }
    let probe: Probe = serde_json::from_slice(bytes).context("parse ffprobe json")?;
    // The video stream is OPTIONAL: audio-only inputs (mp3, the audio-only youtube
    // download) probe fine and report 0×0 so the transcription path can still surface
    // duration + audio. Callers that require pixels (analyze/extract) guard on width>0.
    // Exclude attached-picture streams: embedded cover art (mp3/m4a/flac album art)
    // probes as a "video" stream with real dims but disposition.attached_pic=1 — treat
    // such a file as audio-only, not a 1-frame video, so it routes to transcribe_*.
    let video = probe
        .streams
        .iter()
        .find(|s| s.codec_type.as_deref() == Some("video") && s.disposition.attached_pic == 0);
    let width = video.and_then(|v| v.width).unwrap_or(0);
    let height = video.and_then(|v| v.height).unwrap_or(0);
    let has_audio = probe
        .streams
        .iter()
        .any(|s| s.codec_type.as_deref() == Some("audio"));
    let duration_secs = probe
        .format
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    Ok(VideoMeta {
        width,
        height,
        duration_secs,
        has_audio,
    })
}

/// Parse per-frame `pts_time:<seconds>` values (from ffmpeg `metadata=print` or
/// `showinfo`) in emission order.
fn parse_metadata_pts(text: &str) -> Vec<f64> {
    text.lines()
        .filter_map(|line| {
            let rest = &line[line.find("pts_time:")? + "pts_time:".len()..];
            rest.split_whitespace().next()?.parse::<f64>().ok()
        })
        .collect()
}

/// Reduce `items` to at most `max`, evenly spread, keeping first and last.
/// Returns a clone of all items when already within `max` (or `max == 0`).
fn subsample_even<T: Clone>(items: &[T], max: usize) -> Vec<T> {
    if max == 0 || items.len() <= max {
        return items.to_vec();
    }
    if max == 1 {
        return vec![items[0].clone()];
    }
    let last = (items.len() - 1) as f64;
    (0..max)
        .map(|i| {
            let idx = (i as f64 * last / (max as f64 - 1.0)).round() as usize;
            items[idx.min(items.len() - 1)].clone()
        })
        .collect()
}

/// Fetch duration/dimensions for a remote URL WITHOUT downloading the media
/// (`yt-dlp -j` prints metadata only). Used by the cheap cost estimate.
pub async fn probe_remote(url: &str) -> Result<VideoMeta> {
    let parsed = url::Url::parse(url).with_context(|| format!("invalid URL: {url}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(anyhow!(
            "unsupported URL scheme '{}' — only http/https are allowed",
            parsed.scheme()
        ));
    }
    let out = Command::new("yt-dlp")
        .args(["--no-playlist", "--quiet", "--no-warnings", "-j"])
        .args(ytdlp_extractor_args())
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn yt-dlp -j")?;
    if !out.status.success() {
        return Err(anyhow!(
            "yt-dlp metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse_ytdlp_json(&out.stdout)
}

fn parse_ytdlp_json(bytes: &[u8]) -> Result<VideoMeta> {
    #[derive(serde::Deserialize)]
    struct Info {
        duration: Option<f64>,
        width: Option<u32>,
        height: Option<u32>,
    }
    // yt-dlp -j prints one JSON object (with --no-playlist); take the first line.
    let line = bytes
        .split(|b| *b == b'\n')
        .find(|l| !l.is_empty())
        .unwrap_or(bytes);
    let info: Info = serde_json::from_slice(line).context("parse yt-dlp json")?;
    Ok(VideoMeta {
        width: info.width.unwrap_or(1280),
        height: info.height.unwrap_or(720),
        duration_secs: info.duration.unwrap_or(f64::NAN),
        has_audio: true,
    })
}

/// Tile already-extracted frames into a single contact-sheet JPEG (grid layout).
/// Writes cells to a uuid temp dir, runs ffmpeg `tile`, caps the sheet long edge,
/// returns the JPEG bytes. Temp dir removed on every exit path.
pub async fn make_contact_sheet(
    frames: &[Frame],
    long_edge_cap: u32,
    work_root: &Path,
) -> Result<Vec<u8>> {
    if frames.is_empty() {
        return Err(anyhow!("no frames for contact sheet"));
    }
    tokio::fs::create_dir_all(work_root).await.ok();
    let dir = work_root.join(format!("grid-{}", uuid::Uuid::new_v4()));
    tokio::fs::create_dir_all(&dir)
        .await
        .context("create grid temp dir")?;
    let result = tile_into(frames, long_edge_cap, &dir).await;
    delete_dir_quiet(&dir).await;
    result
}

async fn tile_into(frames: &[Frame], cap: u32, dir: &Path) -> Result<Vec<u8>> {
    for (i, f) in frames.iter().enumerate() {
        tokio::fs::write(dir.join(format!("g_{:05}.jpg", i + 1)), &f.jpeg)
            .await
            .context("write grid cell")?;
    }
    let n = frames.len() as u32;
    let cols = (n as f64).sqrt().ceil() as u32;
    let rows = n.div_ceil(cols.max(1));
    let out = dir.join("sheet.jpg");
    let vf =
        format!("tile={cols}x{rows},scale='min({cap},iw)':'min({cap},ih)':force_original_aspect_ratio=decrease");
    let res = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(dir.join("g_%05d.jpg"))
        .args(["-vf", &vf, "-frames:v", "1", "-q:v", "3"])
        .arg(&out)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .context("spawn ffmpeg (tile)")?;
    if !res.status.success() {
        return Err(anyhow!(
            "ffmpeg tile failed: {}",
            String::from_utf8_lossy(&res.stderr)
        ));
    }
    tokio::fs::read(&out).await.context("read contact sheet")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_args_default_to_the_pot_backed_player_client() {
        let args = build_extractor_args(None, None, None);
        assert_eq!(
            args,
            vec!["--extractor-args", "youtube:player_client=tv_simply"]
        );
    }

    #[test]
    fn extractor_args_add_pot_provider_and_cookies_when_configured() {
        let args = build_extractor_args(
            Some("tv_simply,mweb"),
            Some("http://bgutil-provider:4416"),
            Some("/data/cookies.txt"),
        );
        assert_eq!(
            args,
            vec![
                "--extractor-args",
                "youtube:player_client=tv_simply,mweb",
                "--extractor-args",
                "youtubepot-bgutilhttp:base_url=http://bgutil-provider:4416",
                "--cookies",
                "/data/cookies.txt",
            ]
        );
    }

    #[test]
    fn extractor_args_leave_client_selection_to_ytdlp_on_default_or_blank() {
        assert!(build_extractor_args(Some("default"), None, None).is_empty());
        assert!(build_extractor_args(Some(""), None, None).is_empty());
    }

    #[test]
    fn stderr_tail_keeps_the_last_three_nonblank_lines() {
        let raw = b"warn one\n\nwarn two\nwarn three\nERROR: HTTP Error 403: Forbidden\n\n";
        assert_eq!(
            stderr_tail(raw),
            "warn two; warn three; ERROR: HTTP Error 403: Forbidden"
        );
    }

    #[test]
    fn parses_pts_time_from_metadata_lines() {
        let sample = "frame:0    pts:0       pts_time:0\n\
             lavfi.scene_score=0.000000\n\
             frame:1    pts:512     pts_time:0.512\n\
             lavfi.scene_score=0.421000\n\
             frame:2    pts:1024    pts_time:1.024000\n";
        assert_eq!(parse_metadata_pts(sample), vec![0.0, 0.512, 1.024]);
    }

    #[test]
    fn parse_pts_ignores_non_pts_lines() {
        assert!(parse_metadata_pts("no timestamps here\nkey=value").is_empty());
    }

    #[test]
    fn filtergraph_stamps_metadata_before_print_in_both_modes() {
        // Regression: `fps` (uniform) attaches no frame metadata, so `metadata=print`
        // alone wrote an empty sidecar and every timestamp was lost. The `metadata=add`
        // guard must precede `metadata=print` in BOTH sampling modes.
        let pts = Path::new("/tmp/pts.txt");
        for sampling in [
            SamplingMode::Uniform { fps: 0.5 },
            SamplingMode::Scene { threshold: 0.4 },
        ] {
            let g = build_filtergraph(
                &ExtractParams {
                    box_px: Some(854),
                    sampling,
                    max_frames: 10,
                    dedup: false,
                    start_secs: None,
                    end_secs: None,
                },
                pts,
            );
            let add = g.find("metadata=add").expect("metadata=add present");
            let print = g.find("metadata=print").expect("metadata=print present");
            assert!(add < print, "metadata=add must precede print in {g}");
        }
    }

    #[test]
    fn filtergraph_inserts_mpdecimate_before_print_only_when_dedup() {
        // dedup:true → `mpdecimate` present and ordered before `metadata=print`
        // (it removes frames from the stream, so it must run before the sidecar is
        // written or files.len() would diverge from pts.len()). dedup:false → absent.
        let pts = Path::new("/tmp/pts.txt");
        let base = ExtractParams {
            box_px: Some(1280),
            sampling: SamplingMode::Uniform { fps: 1.0 },
            max_frames: 50,
            dedup: true,
            start_secs: None,
            end_secs: None,
        };
        let on = build_filtergraph(&base, pts);
        let md = on
            .find("mpdecimate")
            .expect("mpdecimate present when dedup");
        let print = on.find("metadata=print").expect("metadata=print present");
        assert!(md < print, "mpdecimate must precede metadata=print in {on}");

        let off = build_filtergraph(
            &ExtractParams {
                dedup: false,
                ..base
            },
            pts,
        );
        assert!(
            !off.contains("mpdecimate"),
            "no mpdecimate when dedup:false in {off}"
        );
    }

    #[test]
    fn window_offset_is_added_to_relative_pts() {
        // A windowed extract seeks with `-ss` (no `-copyts`), so ffmpeg re-bases
        // timestamps to 0 at the seek point. The extractor adds `start_secs` back,
        // so a 0-based sidecar of a [120s, 130s] window reports 120,121,… absolute.
        // This mirrors the exact `.map(|t| t + from)` used in `extract_into`.
        let relative = parse_metadata_pts(
            "frame:0 pts:0 pts_time:0\nframe:1 pts:1 pts_time:1\nframe:2 pts:2 pts_time:2\n",
        );
        let from = 120.0_f64;
        let absolute: Vec<f64> = relative.into_iter().map(|t| t + from).collect();
        assert_eq!(absolute, vec![120.0, 121.0, 122.0]);
    }

    #[test]
    fn subsample_passthrough_when_within_max() {
        let v = vec![1, 2, 3];
        assert_eq!(subsample_even(&v, 5), vec![1, 2, 3]);
        assert_eq!(subsample_even(&v, 3), vec![1, 2, 3]);
    }

    #[test]
    fn subsample_keeps_endpoints_and_spreads() {
        let v: Vec<i32> = (0..=10).collect();
        let s = subsample_even(&v, 3);
        assert_eq!(s, vec![0, 5, 10]);
    }

    #[test]
    fn subsample_max_one() {
        assert_eq!(subsample_even(&[7, 8, 9], 1), vec![7]);
    }

    #[test]
    fn probe_json_parses_dims_duration_audio() {
        let json = br#"{"streams":[{"codec_type":"video","width":1920,"height":1080},{"codec_type":"audio"}],"format":{"duration":"12.5"}}"#;
        let m = parse_probe_json(json).unwrap();
        assert_eq!((m.width, m.height), (1920, 1080));
        assert_eq!(m.duration_secs, 12.5);
        assert!(m.has_audio);
    }

    #[test]
    fn probe_json_no_audio_and_missing_duration() {
        let json = br#"{"streams":[{"codec_type":"video","width":640,"height":480}],"format":{}}"#;
        let m = parse_probe_json(json).unwrap();
        assert!(!m.has_audio);
        assert!(m.duration_secs.is_nan());
    }

    #[test]
    fn probe_json_audio_only_is_zero_dims_not_error() {
        // The transcription path probes audio-only inputs (mp3, the audio-only youtube
        // download). Those have no video stream — parse must still succeed, reporting
        // 0×0 dims but a real duration + audio, so the transcript header shows length.
        let json = br#"{"streams":[{"codec_type":"audio"}],"format":{"duration":"93.2"}}"#;
        let m = parse_probe_json(json).unwrap();
        assert_eq!((m.width, m.height), (0, 0));
        assert_eq!(m.duration_secs, 93.2);
        assert!(m.has_audio);
    }

    #[test]
    fn probe_json_cover_art_counts_as_audio_only() {
        // Embedded album art (mp3/m4a/flac) probes as a "video" stream with real dims
        // but disposition.attached_pic=1. It must NOT be treated as a real video stream,
        // else a podcast gets mislabeled "Video 500×500" and wrongly routed to analyze.
        let json = br#"{"streams":[{"codec_type":"video","width":500,"height":500,"disposition":{"attached_pic":1}},{"codec_type":"audio"}],"format":{"duration":"612"}}"#;
        let m = parse_probe_json(json).unwrap();
        assert_eq!((m.width, m.height), (0, 0));
        assert!(m.has_audio);
        assert_eq!(m.duration_secs, 612.0);
    }
}
