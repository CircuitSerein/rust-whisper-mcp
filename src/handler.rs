use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use rmcp::{
    handler::server::router::tool::ToolRouter,
    model::{
        CallToolResult, ContentBlock, ProgressNotificationParam, ServerCapabilities, ServerInfo,
    },
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router, ErrorData, RoleServer, ServerHandler,
};
use serde::{Deserialize, Serialize};

use crate::{media, paths, whisper::Engine};

#[derive(Clone)]
pub struct WhisperServer {
    engine: Arc<Engine>,
    data_dir: PathBuf,
    /// Optional blended USD-per-1M-tokens rate for cost estimates ($ omitted if None).
    cost_per_mtok: Option<f64>,
    /// Bounds concurrent heavy media work (download + ffmpeg extract) so parallel
    /// analyze calls can't fan out unbounded subprocesses while serialized on the GPU.
    heavy_sem: Arc<tokio::sync::Semaphore>,
    #[allow(dead_code)] // used by #[tool_handler] expansion
    tool_router: ToolRouter<Self>,
}

/// Result returned by the shared transcription helpers.
/// Used as the JSON response body for `/api/transcribe/*` and as the
/// internal value the MCP tool handlers extract `text` from.
#[derive(Debug, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub language: Option<String>,
    pub segments: usize,
    /// Source duration in seconds (ffprobe); `None` when it can't be determined.
    pub duration_secs: Option<f64>,
    /// Video dimensions; `None` for audio-only sources.
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Whether the source carries an audio track.
    pub has_audio: bool,
    /// Timed `[MM:SS]` transcript lines. Always populated in the REST body; the MCP
    /// tool only renders them as text when the caller passes `timestamps:true`.
    pub timed: Vec<TimedLine>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TranscribeFileArgs {
    /// Path to a media file. Accepts container paths (/data/foo.mp4),
    /// Windows host paths (D:\whisper-data\foo.mp4), or filenames under DATA_DIR.
    pub path: String,
    /// Optional ISO-639-1 language code (e.g. "en", "ru"). Auto-detect if omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Irreversibly delete the source file after a successful run. Default: false.
    #[serde(default)]
    pub cleanup: bool,
    /// Render the transcript as timed `[MM:SS]` lines instead of plain text — use when
    /// you need to know WHEN something was said (to cite/seek a moment, or to line the
    /// transcript up with extracted frame windows). Default: false.
    #[serde(default)]
    pub timestamps: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TranscribeYoutubeArgs {
    /// YouTube URL (or any yt-dlp-supported URL).
    pub url: String,
    /// Optional ISO-639-1 language code. Auto-detect if omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Delete the downloaded audio after successful transcription. Default: true.
    #[serde(default = "default_true")]
    pub cleanup: bool,
    /// Render the transcript as timed `[MM:SS]` lines instead of plain text — use when
    /// you need to know WHEN something was said (to cite/seek a moment). Default: false.
    #[serde(default)]
    pub timestamps: bool,
}

fn default_true() -> bool {
    true
}

// ============================= video analysis =============================

/// Default frames per analyze/extract call; hard cap the client can't exceed.
/// A single MCP tool result is bounded by the client's `MAX_MCP_OUTPUT_TOKENS`
/// (~25k default ≈ 20 medium frames), so bigger counts only land when that env is
/// raised or a cheaper quality is used. To cover a WHOLE long video, paginate
/// `extract_frames` by time window (`start_secs`/`end_secs`) — each window is its
/// own call — rather than pushing this cap higher.
const DEFAULT_MAX_FRAMES: u32 = 10;
const MAX_FRAMES_CAP: u32 = 60;
const SCENE_THRESHOLD_DEFAULT: f64 = 0.4;
const SCENE_THRESHOLD_MIN: f64 = 0.1;
const SCENE_THRESHOLD_MAX: f64 = 0.6;
/// Caller-supplied uniform fps is clamped: extraction buffers every emitted frame
/// before subsampling, so an unbounded fps (or fps × a long window) is a disk/memory
/// DoS. Cap the per-second rate AND the total emitted across the window span.
const MAX_UNIFORM_FPS: f64 = 5.0;
const MAX_EMITTED_FRAMES: f64 = 600.0;
/// Anthropic's per-image token estimate divisor (`w*h / 750`).
const PX_PER_TOKEN: u64 = 750;
/// Rough words/min and tokens/word for the transcript estimate. The token factor
/// is conservative (over- rather than under-counting) because non-Latin scripts
/// tokenize heavier than English.
const WORDS_PER_MIN: f64 = 150.0;
const TOKENS_PER_WORD: f64 = 1.8;
/// Total base64 payload ceiling across all returned frames (~24 MB). Frames past
/// it are dropped with a note; a backstop — the frame count cap usually binds first.
const TOTAL_PAYLOAD_CEILING_BYTES: usize = 24 * 1024 * 1024;

fn default_max_frames() -> u32 {
    DEFAULT_MAX_FRAMES
}

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    /// 640px long edge — long video, gist only.
    Compact,
    /// 854px — screen text mostly readable.
    Low,
    /// 1280px — default; screen text clear.
    #[default]
    Medium,
    /// 1568px — the recommended quality/cost ceiling for model ingest.
    High,
}

impl QualityPreset {
    fn box_px(self) -> Option<u32> {
        Some(match self {
            QualityPreset::Compact => 640,
            QualityPreset::Low => 854,
            QualityPreset::Medium => 1280,
            QualityPreset::High => 1568,
        })
    }
    fn label(self) -> &'static str {
        match self {
            QualityPreset::Compact => "compact",
            QualityPreset::Low => "low",
            QualityPreset::Medium => "medium",
            QualityPreset::High => "high",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum Sampling {
    /// Keyframe on each scene change (default).
    #[default]
    Scene,
    /// Fixed frames-per-second.
    Uniform,
}

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// Each frame is its own labelled image block (default).
    #[default]
    Frames,
    /// A single contact-sheet (grid) image — cheap overview, sub-cells lose detail.
    Grid,
}

fn clamp_max_frames(n: u32) -> u32 {
    n.clamp(1, MAX_FRAMES_CAP)
}

fn clamp_scene_threshold(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(SCENE_THRESHOLD_MIN, SCENE_THRESHOLD_MAX)
    } else {
        SCENE_THRESHOLD_DEFAULT
    }
}

/// Downscale-only long-edge cap, mirroring ffmpeg's
/// `force_original_aspect_ratio=decrease`. `None` box = source dims.
fn scaled_dims(w: u32, h: u32, box_px: Option<u32>) -> (u32, u32) {
    match box_px {
        None => (w, h),
        Some(b) => {
            let long = w.max(h);
            if long <= b || long == 0 {
                (w, h)
            } else {
                let s = b as f64 / long as f64;
                (
                    ((w as f64 * s).round() as u32).max(1),
                    ((h as f64 * s).round() as u32).max(1),
                )
            }
        }
    }
}

/// `MM:SS` (minutes may exceed 99 for long videos). Non-finite/negative → `00:00`.
fn fmt_mmss(secs: f64) -> String {
    let total = if secs.is_finite() && secs >= 0.0 {
        secs as u64
    } else {
        0
    };
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Assemble a `TranscribeResult` from whisper output + optional probe metadata.
/// Shared by the file / upload / youtube transcription cores so they stay in sync.
fn transcribe_result(
    out: crate::whisper::TranscriptionOutput,
    meta: Option<&media::VideoMeta>,
    elapsed: std::time::Duration,
) -> TranscribeResult {
    let timed = out
        .timed
        .iter()
        .map(|t| TimedLine {
            mmss: fmt_mmss(t.start_cs as f64 / 100.0),
            text: t.text.clone(),
        })
        .collect();
    TranscribeResult {
        text: out.text,
        language: out.language,
        segments: out.segments,
        duration_secs: meta.map(|m| m.duration_secs).filter(|d| d.is_finite()),
        width: meta.map(|m| m.width).filter(|w| *w > 0),
        height: meta.map(|m| m.height).filter(|h| *h > 0),
        has_audio: meta.map(|m| m.has_audio).unwrap_or(true),
        timed,
        elapsed_ms: elapsed.as_millis(),
    }
}

/// One-line source header prepended to the MCP transcript text, e.g.
/// `Video 1920×1080 · 04:58 · lang en` or `Audio · 03:12 · lang ru`.
fn format_transcribe_header(r: &TranscribeResult) -> String {
    let dur = r
        .duration_secs
        .map(fmt_mmss)
        .unwrap_or_else(|| "??:??".to_string());
    let lang = r.language.as_deref().unwrap_or("?");
    match (r.width, r.height) {
        (Some(w), Some(h)) => {
            let no_audio = if r.has_audio { "" } else { " · no audio" };
            format!("Video {w}×{h} · {dur} · lang {lang}{no_audio}")
        }
        _ => format!("Audio · {dur} · lang {lang}"),
    }
}

/// Transcript body for MCP text output: timed `[MM:SS]` lines when requested (and
/// available), else the plain text.
fn transcribe_body(r: &TranscribeResult, timestamps: bool) -> String {
    if timestamps && !r.timed.is_empty() {
        r.timed
            .iter()
            .map(|l| format!("[{}] {}", l.mmss, l.text))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        r.text.clone()
    }
}

/// Default uniform fps when the caller doesn't set one: aim for ~`max_frames`
/// across the whole video, capped at 1 fps. Falls back to 0.5 fps if duration
/// is unknown.
fn default_uniform_fps(duration_secs: f64, max_frames: u32) -> f64 {
    if duration_secs.is_finite() && duration_secs > 0.0 {
        (max_frames as f64 / duration_secs).clamp(0.01, 1.0)
    } else {
        0.5
    }
}

/// Clamp a caller-supplied uniform fps so extraction can't be driven to buffer an
/// unbounded number of frames. Bounds the per-second rate (`MAX_UNIFORM_FPS`) and the
/// total emitted across the window `span_secs` (`MAX_EMITTED_FRAMES`), lowering the
/// rate rather than truncating so frames stay evenly spaced. Never returns < 0.01.
fn clamp_uniform_fps(requested: f64, span_secs: f64) -> f64 {
    let span_cap = if span_secs.is_finite() && span_secs > 0.0 {
        MAX_EMITTED_FRAMES / span_secs
    } else {
        MAX_UNIFORM_FPS
    };
    let hi = MAX_UNIFORM_FPS.min(span_cap).max(0.01);
    requested.clamp(0.01, hi)
}

/// Effective span of the extraction window (seconds), used to pick the uniform
/// fallback fps. Falls back to the full duration when no window is set, so a
/// windowed extract that hits the no-scene-cut fallback spreads frames over the
/// window, not the whole video.
fn window_span(start: Option<f64>, end: Option<f64>, full: f64) -> f64 {
    let from = start.filter(|s| s.is_finite() && *s > 0.0).unwrap_or(0.0);
    match end.filter(|e| e.is_finite() && *e > from) {
        Some(e) => e - from,
        None if full.is_finite() && full > from => full - from,
        None => full,
    }
}

fn resolve_sampling(
    sampling: Sampling,
    scene_threshold: Option<f64>,
    fps: Option<f64>,
    duration_secs: f64,
    max_frames: u32,
) -> media::SamplingMode {
    match sampling {
        Sampling::Scene => media::SamplingMode::Scene {
            threshold: clamp_scene_threshold(scene_threshold.unwrap_or(SCENE_THRESHOLD_DEFAULT)),
        },
        Sampling::Uniform => {
            let fps = fps
                .filter(|f| f.is_finite() && *f > 0.0)
                .map(|f| clamp_uniform_fps(f, duration_secs))
                .unwrap_or_else(|| default_uniform_fps(duration_secs, max_frames));
            media::SamplingMode::Uniform { fps }
        }
    }
}

/// Estimate the token cost of a set of frames + transcript. Conservative upper
/// bound (frames are ≤1568px so no host resizes them; transcript factor over-counts).
#[allow(clippy::too_many_arguments)]
fn estimate_tokens(
    src_w: u32,
    src_h: u32,
    duration_secs: f64,
    frame_count: usize,
    box_px: Option<u32>,
    max_frames: u32,
    cost_per_mtok: Option<f64>,
) -> TokenEstimate {
    let (w, h) = scaled_dims(src_w, src_h, box_px);
    let tokens_per_frame = ((w as u64) * (h as u64)).div_ceil(PX_PER_TOKEN);
    let n = (frame_count as u32).min(max_frames) as u64;
    let visual_tokens = tokens_per_frame * n;
    let transcript_tokens = if duration_secs.is_finite() && duration_secs > 0.0 {
        ((duration_secs / 60.0) * WORDS_PER_MIN * TOKENS_PER_WORD).ceil() as u64
    } else {
        0
    };
    let total_tokens = visual_tokens + transcript_tokens;
    let usd = cost_per_mtok.map(|r| total_tokens as f64 / 1_000_000.0 * r);
    TokenEstimate {
        frames: n as u32,
        frame_width: w,
        frame_height: h,
        tokens_per_frame,
        visual_tokens,
        transcript_tokens,
        total_tokens,
        usd,
        note: "rough upper bound, ±50%; non-Latin transcripts may run higher".to_string(),
    }
}

#[derive(Debug, Serialize)]
pub struct TokenEstimate {
    pub frames: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub tokens_per_frame: u64,
    pub visual_tokens: u64,
    pub transcript_tokens: u64,
    pub total_tokens: u64,
    pub usd: Option<f64>,
    pub note: String,
}

#[derive(Debug, Serialize)]
pub struct FrameOut {
    pub index: usize,
    pub timestamp_secs: f64,
    pub mmss: String,
    pub jpeg_base64: String,
}

#[derive(Debug, Serialize)]
pub struct TimedLine {
    pub mmss: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeResult {
    pub width: u32,
    pub height: u32,
    pub duration_secs: f64,
    pub sampling_used: String,
    pub quality: String,
    pub language: Option<String>,
    pub full_text: String,
    pub frames: Vec<FrameOut>,
    pub transcript: Vec<TimedLine>,
    pub estimate: TokenEstimate,
    pub notes: Vec<String>,
    pub elapsed_ms: u128,
}

/// Normalized analyze options shared by file/youtube/extract cores.
struct AnalyzeOpts {
    quality: QualityPreset,
    sampling: Sampling,
    scene_threshold: Option<f64>,
    fps: Option<f64>,
    max_frames: u32,
    dedup: bool,
    layout: Layout,
    language: Option<String>,
    transcribe: bool,
    /// Optional [start_secs, end_secs] extraction window (extract_frames pagination).
    /// `None` on analyze_file/youtube — those are whole-clip, single-call tools.
    start_secs: Option<f64>,
    end_secs: Option<f64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnalyzeFileArgs {
    /// Path to a video file. Resolved under DATA_DIR, same rules as transcribe_file.
    pub path: String,
    /// Optional ISO-639-1 language code (e.g. "en", "ru"). Auto-detect if omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Irreversibly delete the source file after analysis. Default: false.
    #[serde(default)]
    pub cleanup: bool,
    /// Frame resolution preset: compact|low|medium|high. Default: medium (1280px).
    #[serde(default)]
    pub quality: QualityPreset,
    /// Frame sampling: scene (default) or uniform.
    #[serde(default)]
    pub sampling: Sampling,
    /// Scene-change sensitivity 0.1–0.6 (lower = more frames). Default 0.4.
    #[serde(default)]
    pub scene_threshold: Option<f64>,
    /// Frames per second when sampling=uniform (clamped to a sane max). Default:
    /// derived from duration to spread ~max_frames evenly.
    #[serde(default)]
    pub fps: Option<f64>,
    /// Upper bound on frames returned (dedup may yield fewer). Hard cap 60, but one
    /// result is token-capped to ~20 medium frames in practice — for a long video,
    /// page extract_frames by window rather than raising this. Default 10.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Result layout: frames (default, one labelled image per frame) or grid (single
    /// contact sheet — cheap overview, sub-cells lose detail).
    #[serde(default)]
    pub layout: Layout,
    /// Drop near-identical consecutive frames (ffmpeg mpdecimate) so static runs
    /// don't return duplicate images. Default: true.
    #[serde(default = "default_true")]
    pub dedup: bool,
}

impl AnalyzeFileArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            dedup: self.dedup,
            layout: self.layout,
            language: self.language.clone(),
            transcribe: true,
            start_secs: None,
            end_secs: None,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnalyzeYoutubeArgs {
    /// YouTube (or any yt-dlp-supported) URL. Video+audio downloaded once.
    pub url: String,
    /// Optional ISO-639-1 language code (e.g. "en", "ru"). Auto-detect if omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Delete the downloaded video after analysis. Default: true.
    #[serde(default = "default_true")]
    pub cleanup: bool,
    /// Frame resolution: compact (640px) | low (854) | medium (1280) | high (1568).
    /// Higher = clearer on-screen text but more tokens. Default: medium.
    #[serde(default)]
    pub quality: QualityPreset,
    /// Frame sampling: scene (keyframe per scene change, default) or uniform (fixed fps).
    #[serde(default)]
    pub sampling: Sampling,
    /// Scene-change sensitivity 0.1–0.6 (lower = more frames). Only used when
    /// sampling=scene. Default 0.4.
    #[serde(default)]
    pub scene_threshold: Option<f64>,
    /// Frames per second when sampling=uniform (clamped to a sane max). Default:
    /// derived from duration to spread ~max_frames evenly.
    #[serde(default)]
    pub fps: Option<f64>,
    /// Upper bound on frames returned (dedup may yield fewer). Hard cap 60, but one
    /// result is token-capped to ~20 medium frames in practice — for a long video,
    /// page extract_frames by window rather than raising this. Default 10.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Result layout: frames (default, one labelled image per frame) or grid (single
    /// contact sheet — cheap overview, sub-cells lose detail).
    #[serde(default)]
    pub layout: Layout,
    /// Drop near-identical consecutive frames (ffmpeg mpdecimate). Default: true.
    #[serde(default = "default_true")]
    pub dedup: bool,
}

impl AnalyzeYoutubeArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            dedup: self.dedup,
            layout: self.layout,
            language: self.language.clone(),
            transcribe: true,
            start_secs: None,
            end_secs: None,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExtractFramesArgs {
    /// Path to a video file (resolved under DATA_DIR).
    pub path: String,
    /// Frame resolution: compact (640px) | low (854) | medium (1280) | high (1568).
    /// Higher = clearer on-screen text but more tokens. Default: medium.
    #[serde(default)]
    pub quality: QualityPreset,
    /// Frame sampling: scene (keyframe per scene change, default) or uniform (fixed fps).
    #[serde(default)]
    pub sampling: Sampling,
    /// Scene-change sensitivity 0.1–0.6 (lower = more frames). Only used when
    /// sampling=scene. Default 0.4.
    #[serde(default)]
    pub scene_threshold: Option<f64>,
    /// Frames per second when sampling=uniform (clamped to a sane max). Default:
    /// derived from the window duration to spread ~max_frames evenly.
    #[serde(default)]
    pub fps: Option<f64>,
    /// Upper bound on frames returned (dedup may yield fewer). Hard cap 60, but one
    /// result is token-capped to ~20 medium frames in practice. Default 10.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Result layout: frames (default, one labelled image per frame) or grid (single
    /// contact sheet — cheap overview, sub-cells lose detail).
    #[serde(default)]
    pub layout: Layout,
    /// Extract only frames from this source time onward (seconds). Pair with
    /// `end_secs` to page through a long video one window at a time — frame
    /// timestamps are reported as absolute [MM:SS]. Default: from the start.
    #[serde(default)]
    pub start_secs: Option<f64>,
    /// Extract only frames up to this source time (seconds). Default: to the end.
    #[serde(default)]
    pub end_secs: Option<f64>,
    /// Irreversibly delete the source file after extraction. Default: false.
    #[serde(default)]
    pub cleanup: bool,
    /// Drop near-identical consecutive frames (ffmpeg mpdecimate) so static runs
    /// collapse to one image. Makes `max_frames` an upper bound. Default: true.
    #[serde(default = "default_true")]
    pub dedup: bool,
}

impl ExtractFramesArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            dedup: self.dedup,
            layout: self.layout,
            language: None,
            transcribe: false,
            start_secs: self.start_secs,
            end_secs: self.end_secs,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EstimateArgs {
    /// Local video path (resolved under DATA_DIR). Provide this OR `url`.
    #[serde(default)]
    pub path: Option<String>,
    /// Remote URL. Metadata is fetched via yt-dlp without downloading.
    #[serde(default)]
    pub url: Option<String>,
    /// Frame resolution the estimate assumes: compact (640px) | low (854) | medium
    /// (1280) | high (1568). Default: medium.
    #[serde(default)]
    pub quality: QualityPreset,
    /// scene (upper-bound frame count = max_frames) or uniform (duration × fps).
    #[serde(default)]
    pub sampling: Sampling,
    /// Frames per second for uniform sampling. Default derived from duration.
    #[serde(default)]
    pub fps: Option<f64>,
    /// Frame count the estimate assumes (upper bound; hard cap 60). Default 10.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
}

#[tool_router]
impl WhisperServer {
    pub fn new(engine: Arc<Engine>, data_dir: PathBuf, cost_per_mtok: Option<f64>) -> Self {
        Self {
            engine,
            data_dir,
            cost_per_mtok,
            heavy_sem: Arc::new(tokio::sync::Semaphore::new(2)),
            tool_router: Self::tool_router(),
        }
    }

    /// Shared core: resolve path, decode, transcribe, optionally cleanup.
    /// Called by both the MCP tool method and the REST handler.
    pub async fn run_file<F>(
        &self,
        path: &str,
        language: Option<&str>,
        cleanup: bool,
        on_whisper_progress: F,
    ) -> Result<TranscribeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let started = Instant::now();
        let req_id = short_id();
        let span = tracing::info_span!("transcribe_file", req = %req_id);
        let _enter = span.enter();
        tracing::info!(path, language = ?language, cleanup, "request received");

        let resolved = paths::resolve_in_data_dir(path, &self.data_dir).context("resolve path")?;

        // Cheap ffprobe for duration/dimensions/audio so the transcript carries basic
        // info without the caller reaching for the heavy image path. Failure is
        // non-fatal — transcription still proceeds with metadata omitted — but log it,
        // else a real video whose probe failed silently reports as "Audio · ??:??".
        let meta = media::probe_video(&resolved)
            .await
            .inspect_err(|e| {
                tracing::warn!(path = %resolved.display(), "probe_video failed: {e:#}; header omits duration/dimensions")
            })
            .ok();

        let pcm = media::decode_to_pcm(&resolved)
            .await
            .context("ffmpeg decode")?;
        let out = self
            .transcribe_pcm_async(pcm, language, on_whisper_progress)
            .await
            .context("whisper")?;

        if cleanup {
            media::delete_quiet(&resolved).await;
        }

        let elapsed = started.elapsed();
        tracing::info!(
            lang = out.language.as_deref().unwrap_or("?"),
            segments = out.segments,
            chars = out.text.len(),
            total_ms = elapsed.as_millis() as u64,
            "transcribe_file complete"
        );

        Ok(transcribe_result(out, meta.as_ref(), elapsed))
    }

    /// Write uploaded bytes to an inbox under DATA_DIR, transcribe, then delete.
    /// Used by the REST upload endpoint when the host file lives outside DATA_DIR.
    pub async fn run_upload<F>(
        &self,
        bytes: Vec<u8>,
        original_name: Option<&str>,
        language: Option<&str>,
        on_whisper_progress: F,
    ) -> Result<TranscribeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let started = Instant::now();
        let req_id = short_id();
        let span = tracing::info_span!("transcribe_upload", req = %req_id);
        let _enter = span.enter();

        let ext = original_name
            .and_then(|n| std::path::Path::new(n).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("bin");
        let id = uuid::Uuid::new_v4();
        let inbox = self.data_dir.join("__inbox__");
        tokio::fs::create_dir_all(&inbox)
            .await
            .context("create inbox dir")?;
        let path = inbox.join(format!("{id}.{ext}"));
        let size = bytes.len();
        tokio::fs::write(&path, bytes)
            .await
            .context("write upload")?;
        tracing::info!(
            path = %path.display(),
            bytes = size,
            original = original_name.unwrap_or("?"),
            language = ?language,
            "upload received"
        );

        let meta = media::probe_video(&path)
            .await
            .inspect_err(|e| {
                tracing::warn!(path = %path.display(), "probe_video failed: {e:#}; header omits duration/dimensions")
            })
            .ok();
        let pcm = media::decode_to_pcm(&path).await.context("ffmpeg decode")?;
        let out = self
            .transcribe_pcm_async(pcm, language, on_whisper_progress)
            .await
            .context("whisper")?;

        media::delete_quiet(&path).await;

        let elapsed = started.elapsed();
        tracing::info!(
            lang = out.language.as_deref().unwrap_or("?"),
            segments = out.segments,
            chars = out.text.len(),
            total_ms = elapsed.as_millis() as u64,
            "transcribe_upload complete"
        );

        Ok(transcribe_result(out, meta.as_ref(), elapsed))
    }

    /// Shared core: download via yt-dlp, decode, transcribe, optionally cleanup.
    pub async fn run_youtube<F>(
        &self,
        url: &str,
        language: Option<&str>,
        cleanup: bool,
        on_whisper_progress: F,
    ) -> Result<TranscribeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let started = Instant::now();
        let req_id = short_id();
        let span = tracing::info_span!("transcribe_youtube", req = %req_id);
        let _enter = span.enter();
        tracing::info!(url, language = ?language, cleanup, "request received");

        let yt_dir = self.data_dir.join("youtube");
        let file = media::download_youtube(url, &yt_dir)
            .await
            .context("yt-dlp")?;

        let meta = media::probe_video(&file)
            .await
            .inspect_err(|e| {
                tracing::warn!(path = %file.display(), "probe_video failed: {e:#}; header omits duration/dimensions")
            })
            .ok();
        let pcm = media::decode_to_pcm(&file).await.context("ffmpeg decode")?;
        let out = self
            .transcribe_pcm_async(pcm, language, on_whisper_progress)
            .await
            .context("whisper")?;

        if cleanup {
            media::delete_quiet(&file).await;
        }

        let elapsed = started.elapsed();
        tracing::info!(
            lang = out.language.as_deref().unwrap_or("?"),
            segments = out.segments,
            chars = out.text.len(),
            total_ms = elapsed.as_millis() as u64,
            "transcribe_youtube complete"
        );

        Ok(transcribe_result(out, meta.as_ref(), elapsed))
    }

    async fn transcribe_pcm_async<F>(
        &self,
        pcm: Vec<f32>,
        language: Option<&str>,
        on_progress: F,
    ) -> Result<crate::whisper::TranscriptionOutput>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let started = Instant::now();
        let engine = self.engine.clone();
        let lang = language.map(|s| s.to_string());
        tracing::info!("whisper inference starting");
        let out = tokio::task::spawn_blocking(move || {
            engine.transcribe_pcm(&pcm, lang.as_deref(), on_progress)
        })
        .await
        .context("join whisper task")??;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis() as u64,
            "whisper inference complete"
        );
        Ok(out)
    }

    #[tool(
        name = "transcribe_file",
        description = "Transcribe a local media file (any format ffmpeg understands) with Whisper large-v3 on GPU, returning the transcript plus a one-line info header (duration, resolution, language). NO images — this is the light default for 'what does it say?'. Pass timestamps:true for [MM:SS] lines that anchor frame windows. Path is resolved under the container's DATA_DIR mount."
    )]
    async fn transcribe_file(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            TranscribeFileArgs,
        >,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        emit_progress(&ctx, 0.05, Some("decoding audio".into())).await;
        let cb = whisper_progress_callback(&ctx);
        let res = self
            .run_file(&args.path, args.language.as_deref(), args.cleanup, cb)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        let text = format!(
            "{}\n{}",
            format_transcribe_header(&res),
            transcribe_body(&res, args.timestamps)
        );
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    #[tool(
        name = "transcribe_youtube",
        description = "Download a YouTube (or any yt-dlp-supported) URL as audio and transcribe with Whisper large-v3, returning the transcript plus a one-line info header (duration, language). NO images — the light default. Pass timestamps:true for [MM:SS] lines. The download lands under DATA_DIR and is cleaned up by default."
    )]
    async fn transcribe_youtube(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            TranscribeYoutubeArgs,
        >,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        emit_progress(&ctx, 0.02, Some("downloading from youtube".into())).await;
        let cb = whisper_progress_callback(&ctx);
        let res = self
            .run_youtube(&args.url, args.language.as_deref(), args.cleanup, cb)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        let text = format!(
            "{}\n{}",
            format_transcribe_header(&res),
            transcribe_body(&res, args.timestamps)
        );
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    // ---------------- video analysis cores (shared by MCP + REST) ----------------

    /// Cheap token/cost estimate: ffprobe (local) or yt-dlp metadata (remote) plus
    /// arithmetic. No decode, no frame writes, no whisper, no images returned.
    /// Scene mode uses `max_frames` as the upper-bound frame count.
    pub async fn run_estimate(&self, args: &EstimateArgs) -> Result<TokenEstimate> {
        let max_frames = clamp_max_frames(args.max_frames);
        let box_px = args.quality.box_px();
        let (w, h, duration) = match (&args.path, &args.url) {
            (Some(path), None) => {
                let file =
                    paths::resolve_in_data_dir(path, &self.data_dir).context("resolve path")?;
                let m = media::probe_video(&file).await.context("probe video")?;
                (m.width, m.height, m.duration_secs)
            }
            (None, Some(url)) => {
                let m = media::probe_remote(url).await.context("probe remote")?;
                (m.width, m.height, m.duration_secs)
            }
            (Some(_), Some(_)) => return Err(anyhow::anyhow!("provide `path` OR `url`, not both")),
            (None, None) => return Err(anyhow::anyhow!("provide one of `path` or `url`")),
        };
        // Mirror analyze_common's guard: an audio-only source (0×0) can't be analyzed,
        // so don't hand back a bogus 0-visual-token estimate that the very next
        // analyze_file call would then reject.
        if w == 0 || h == 0 {
            return Err(anyhow::anyhow!(
                "no video stream — use transcribe_file for audio-only input"
            ));
        }
        let frame_count = match args.sampling {
            Sampling::Scene => max_frames as usize,
            Sampling::Uniform => {
                let fps = args
                    .fps
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .unwrap_or_else(|| default_uniform_fps(duration, max_frames));
                if duration.is_finite() && duration > 0.0 {
                    ((duration * fps).ceil() as usize).min(max_frames as usize)
                } else {
                    max_frames as usize
                }
            }
        };
        Ok(estimate_tokens(
            w,
            h,
            duration,
            frame_count,
            box_px,
            max_frames,
            self.cost_per_mtok,
        ))
    }

    pub async fn run_analyze_file<F>(
        &self,
        args: &AnalyzeFileArgs,
        on_progress: F,
    ) -> Result<AnalyzeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let _permit = self.heavy_sem.clone().acquire_owned().await.ok();
        let file =
            paths::resolve_in_data_dir(&args.path, &self.data_dir).context("resolve path")?;
        let res = self.analyze_common(&file, args.opts(), on_progress).await?;
        if args.cleanup {
            media::delete_quiet(&file).await;
        }
        Ok(res)
    }

    pub async fn run_analyze_youtube<F>(
        &self,
        args: &AnalyzeYoutubeArgs,
        on_progress: F,
    ) -> Result<AnalyzeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let _permit = self.heavy_sem.clone().acquire_owned().await.ok();
        let yt_dir = self.data_dir.join("youtube");
        let file = media::download_video(&args.url, &yt_dir)
            .await
            .context("yt-dlp")?;
        let mut res = self.analyze_common(&file, args.opts(), on_progress).await;
        if args.cleanup {
            media::delete_quiet(&file).await;
        } else if let Ok(r) = res.as_mut() {
            // Keep the download and surface its DATA_DIR-relative path so a caller
            // can page more windows over it with extract_frames / transcribe_file
            // (and delete it via a final extract_frames call with cleanup:true).
            if let Ok(rel) = file.strip_prefix(&self.data_dir) {
                r.notes.push(format!(
                    "[downloaded video kept at {} — page more detail with \
                     extract_frames(path, start_secs, end_secs); pass cleanup:true on \
                     the last call to delete it]",
                    rel.display()
                ));
            }
        }
        res
    }

    pub async fn run_extract_frames(&self, args: &ExtractFramesArgs) -> Result<AnalyzeResult> {
        let _permit = self.heavy_sem.clone().acquire_owned().await.ok();
        let file =
            paths::resolve_in_data_dir(&args.path, &self.data_dir).context("resolve path")?;
        let res = self.analyze_common(&file, args.opts(), |_| {}).await?;
        if args.cleanup {
            media::delete_quiet(&file).await;
        }
        Ok(res)
    }

    /// Shared analyze pipeline: probe → (whisper ∥ keyframe extraction) → assemble.
    async fn analyze_common<F>(
        &self,
        file: &Path,
        opts: AnalyzeOpts,
        on_progress: F,
    ) -> Result<AnalyzeResult>
    where
        F: Fn(i32) + Send + Sync + 'static,
    {
        let started = Instant::now();
        let mut notes: Vec<String> = Vec::new();

        let meta = media::probe_video(file).await.context("probe video")?;

        // Analyze/extract need a video stream; audio-only inputs (0×0) can't yield
        // frames, so fail cleanly here instead of cryptically at ffmpeg. Transcription
        // handles audio-only fine via the transcribe_* tools.
        if meta.width == 0 || meta.height == 0 {
            return Err(anyhow::anyhow!(
                "no video stream — use transcribe_file for audio-only input"
            ));
        }

        // Validate the extraction window up front: a bad window should error cheaply,
        // not silently re-extract to EOF (which would surprise a pagination loop).
        // end_secs must exceed the effective start (0 when start is unset), else the
        // window silently collapses to "no bound" downstream (window_span / the `-t`
        // computation drop a non-positive end) and extracts to EOF — surprising a
        // pagination loop. Validate it even when start_secs is absent.
        let start0 = opts
            .start_secs
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(0.0);
        if let Some(e) = opts.end_secs {
            if !e.is_finite() || e <= start0 {
                return Err(anyhow::anyhow!(
                    "invalid window: end_secs ({e}) must be greater than start_secs ({start0})"
                ));
            }
        }
        if let Some(s) = opts.start_secs {
            if s < 0.0 || (meta.duration_secs.is_finite() && s >= meta.duration_secs) {
                return Err(anyhow::anyhow!(
                    "invalid window: start_secs ({s}) is out of range for a {:.3}s video",
                    meta.duration_secs
                ));
            }
        }

        let box_px = opts.quality.box_px();
        // Uniform fps (when the caller gives none) must be derived from the WINDOW
        // span, not the whole video, or a windowed uniform extract samples far too
        // sparsely. `window_span` == full duration when no window is set.
        let span = window_span(opts.start_secs, opts.end_secs, meta.duration_secs);
        let sampling = resolve_sampling(
            opts.sampling,
            opts.scene_threshold,
            opts.fps,
            span,
            opts.max_frames,
        );
        let params = media::ExtractParams {
            box_px,
            sampling,
            max_frames: opts.max_frames,
            dedup: opts.dedup,
            start_secs: opts.start_secs,
            end_secs: opts.end_secs,
        };
        let frames_dir = self.data_dir.join("frames");

        // Overlap GPU-bound whisper (if any audio) with decode + keyframe extraction.
        let want_audio = opts.transcribe && meta.has_audio;
        let lang = opts.language.clone();
        let engine_fut = async {
            if want_audio {
                let pcm = media::decode_to_pcm(file).await.context("ffmpeg decode")?;
                let out = self
                    .transcribe_pcm_async(pcm, lang.as_deref(), on_progress)
                    .await
                    .context("whisper")?;
                Ok::<Option<crate::whisper::TranscriptionOutput>, anyhow::Error>(Some(out))
            } else {
                Ok(None)
            }
        };
        let extract_fut = media::extract_keyframes(file, &frames_dir, &params);
        let (whisper_out, mut frames) = tokio::try_join!(engine_fut, extract_fut)?;

        if opts.transcribe && !meta.has_audio {
            notes.push("[no audio track — transcript empty]".to_string());
        }

        let mut sampling_used = describe_sampling(sampling);
        // Zero scene cuts (static screencast) → uniform fallback so frames still return.
        if frames.is_empty() {
            let fps = default_uniform_fps(span, opts.max_frames);
            let uni = media::ExtractParams {
                box_px,
                sampling: media::SamplingMode::Uniform { fps },
                max_frames: opts.max_frames,
                dedup: opts.dedup,
                start_secs: opts.start_secs,
                end_secs: opts.end_secs,
            };
            frames = media::extract_keyframes(file, &frames_dir, &uni)
                .await
                .context("uniform fallback extract")?;
            if frames.is_empty() {
                // Genuinely nothing here (e.g. a window past the end of the video) —
                // don't claim uniform sampling happened.
                notes.push("[no frames in the requested range]".to_string());
            } else {
                notes.push(format!(
                    "[no scene cuts detected — sampled uniformly at {fps:.3} fps]"
                ));
                sampling_used = format!("scene→uniform(fps={fps:.3})");
            }
        }

        // Duration: prefer ffprobe; else last frame timestamp as a lower bound.
        let duration_secs = if meta.duration_secs.is_finite() {
            meta.duration_secs
        } else {
            frames.last().map(|f| f.pts_time).unwrap_or(0.0)
        };

        let (full_text, language, timed) = match whisper_out {
            Some(o) => (o.text, o.language, o.timed),
            None => (String::new(), None, Vec::new()),
        };
        let transcript: Vec<TimedLine> = timed
            .iter()
            .map(|t| TimedLine {
                mmss: fmt_mmss(t.start_cs as f64 / 100.0),
                text: t.text.clone(),
            })
            .collect();

        // Grid: collapse frames into one contact-sheet image (cell→time map in notes).
        let frames_out = if matches!(opts.layout, Layout::Grid) && !frames.is_empty() {
            let cap = box_px.unwrap_or(1568).max(640);
            let cells: Vec<String> = frames
                .iter()
                .map(|f| format!("{}={}", f.index, fmt_mmss(f.pts_time)))
                .collect();
            notes.push(format!("[grid cells row-major: {}]", cells.join(", ")));
            let sheet = media::make_contact_sheet(&frames, cap, &frames_dir)
                .await
                .context("contact sheet")?;
            use base64::Engine as _;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&sheet);
            vec![FrameOut {
                index: 1,
                timestamp_secs: 0.0,
                mmss: "grid".to_string(),
                jpeg_base64: b64,
            }]
        } else {
            let (outs, dropped) = encode_frames_capped(frames).await;
            if dropped > 0 {
                notes.push(format!(
                    "[frames truncated: {} of {} shown — payload cap]",
                    outs.len(),
                    outs.len() + dropped
                ));
            }
            outs
        };

        let estimate = estimate_tokens(
            meta.width,
            meta.height,
            duration_secs,
            frames_out.len(),
            box_px,
            opts.max_frames,
            self.cost_per_mtok,
        );

        Ok(AnalyzeResult {
            width: meta.width,
            height: meta.height,
            duration_secs,
            sampling_used,
            quality: opts.quality.label().to_string(),
            language,
            full_text,
            frames: frames_out,
            transcript,
            estimate,
            notes,
            elapsed_ms: started.elapsed().as_millis(),
        })
    }

    #[tool(
        name = "estimate_video_cost",
        description = "Preview the token cost (and USD if a rate is configured) of a frame extraction WITHOUT running whisper or returning images — fast, it only probes metadata. Returns per-frame and total token counts. Use before analyze_file/analyze_youtube, or to size max_frames / a window for extract_frames. Provide path OR url, not both; audio-only input errors."
    )]
    async fn estimate_video_cost(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            EstimateArgs,
        >,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let est = self
            .run_estimate(&args)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(
            format_estimate(&est),
        )]))
    }

    #[tool(
        name = "analyze_file",
        description = "See a SHORT local video (roughly under ~2 min / a handful of scenes): extract keyframes AND transcribe in one call, returning frames as images interleaved with a timestamped transcript so a vision model can 'see' it. Video only — audio-only input errors; use transcribe_file for that. For anything longer, prefer transcribe_file once + extract_frames per window (images are ~1.2k tokens each at medium and accumulate). Path resolved under DATA_DIR."
    )]
    async fn analyze_file(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            AnalyzeFileArgs,
        >,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        emit_progress(&ctx, 0.05, Some("probing video".into())).await;
        let layout = args.layout;
        let cb = whisper_progress_callback(&ctx);
        let res = self
            .run_analyze_file(&args, cb)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        Ok(CallToolResult::success(build_content(&res, layout)))
    }

    #[tool(
        name = "analyze_youtube",
        description = "See a SHORT video URL (roughly under ~2 min): download once (video+audio), extract keyframes AND transcribe, returning frames as images plus a timestamped transcript. Video only — audio-only input errors. For anything longer, prefer one transcript pass + extract_frames per window; cleanup:false keeps the download for that paging. Cleaned up by default."
    )]
    async fn analyze_youtube(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            AnalyzeYoutubeArgs,
        >,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        emit_progress(&ctx, 0.02, Some("downloading video".into())).await;
        let layout = args.layout;
        let cb = whisper_progress_callback(&ctx);
        let res = self
            .run_analyze_youtube(&args, cb)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        Ok(CallToolResult::success(build_content(&res, layout)))
    }

    #[tool(
        name = "extract_frames",
        description = "Extract video keyframes only (no transcription), returning them as images. Near-identical consecutive frames are dropped automatically (dedup — so max_frames is an upper bound). Same sampling/quality options as analyze_file, plus an optional [start_secs, end_secs] window: page through a long video one window at a time (frame timestamps stay absolute [MM:SS]). Pair with one transcribe_file(timestamps:true) call to analyze a whole video across many calls."
    )]
    async fn extract_frames(
        &self,
        rmcp::handler::server::wrapper::Parameters(args): rmcp::handler::server::wrapper::Parameters<
            ExtractFramesArgs,
        >,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let layout = args.layout;
        let res = self
            .run_extract_frames(&args)
            .await
            .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))?;
        Ok(CallToolResult::success(build_content(&res, layout)))
    }
}

fn short_id() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    id[..8].to_string()
}

fn describe_sampling(s: media::SamplingMode) -> String {
    match s {
        media::SamplingMode::Scene { threshold } => format!("scene(threshold={threshold})"),
        media::SamplingMode::Uniform { fps } => format!("uniform(fps={fps})"),
    }
}

/// base64-encode frames on a blocking thread (CPU-bound over tens of MB), stopping
/// once the cumulative payload would exceed the ceiling. Returns (encoded, dropped).
async fn encode_frames_capped(frames: Vec<media::Frame>) -> (Vec<FrameOut>, usize) {
    tokio::task::spawn_blocking(move || {
        use base64::Engine as _;
        let total_in = frames.len();
        let mut out: Vec<FrameOut> = Vec::with_capacity(total_in);
        let mut total = 0usize;
        for f in frames {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&f.jpeg);
            if !out.is_empty() && total + b64.len() > TOTAL_PAYLOAD_CEILING_BYTES {
                break;
            }
            total += b64.len();
            out.push(FrameOut {
                index: out.len() + 1,
                timestamp_secs: f.pts_time,
                mmss: fmt_mmss(f.pts_time),
                jpeg_base64: b64,
            });
        }
        let dropped = total_in - out.len();
        (out, dropped)
    })
    .await
    .unwrap_or_else(|_| (Vec::new(), 0))
}

fn format_estimate(e: &TokenEstimate) -> String {
    let usd = e
        .usd
        .map(|u| format!("\n≈ ${u:.2} at the configured rate"))
        .unwrap_or_default();
    format!(
        "~{} frames @ {}×{} ≈ {} visual tokens\n+ ~{} transcript ≈ {} tokens total{}\n({})",
        e.frames,
        e.frame_width,
        e.frame_height,
        e.visual_tokens,
        e.transcript_tokens,
        e.total_tokens,
        usd,
        e.note
    )
}

fn format_metadata(r: &AnalyzeResult) -> String {
    let lang = r.language.as_deref().unwrap_or("?");
    let mut s = format!(
        "Video analysis: {}×{}, {} duration, {} frame(s) @ {}, quality {}, language {}. ~{} tokens (est). {} ms.",
        r.width,
        r.height,
        fmt_mmss(r.duration_secs),
        r.frames.len(),
        r.sampling_used,
        r.quality,
        lang,
        r.estimate.total_tokens,
        r.elapsed_ms
    );
    for n in &r.notes {
        s.push('\n');
        s.push_str(n);
    }
    s
}

/// Assemble the mixed text+image tool result: metadata → timestamped transcript →
/// frames (per-frame label + image for `frames` layout, or the single grid image).
///
/// Transcript comes BEFORE the frames on purpose: a single MCP tool result is
/// capped (`MAX_MCP_OUTPUT_TOKENS`, ~25k default) and the client truncates the
/// tail. The transcript is the cheap, always-useful part, so it must survive; the
/// expensive frames are the tail that gets cut first if anything does.
fn build_content(r: &AnalyzeResult, layout: Layout) -> Vec<ContentBlock> {
    let mut blocks = vec![ContentBlock::text(format_metadata(r))];
    if !r.transcript.is_empty() {
        let body = r
            .transcript
            .iter()
            .map(|l| format!("[{}] {}", l.mmss, l.text))
            .collect::<Vec<_>>()
            .join("\n");
        blocks.push(ContentBlock::text(format!("Transcript:\n{body}")));
    }
    for f in &r.frames {
        if matches!(layout, Layout::Frames) {
            blocks.push(ContentBlock::text(format!(
                "[frame {}] {}",
                f.index, f.mmss
            )));
        }
        blocks.push(ContentBlock::image(f.jpeg_base64.clone(), "image/jpeg"));
    }
    blocks
}

/// Build a closure that forwards whisper's 0..100 progress ticks to the
/// MCP client as `notify_progress` calls. The mapping puts whisper into
/// the 0.20..0.98 range of the overall task.
fn whisper_progress_callback(
    ctx: &RequestContext<RoleServer>,
) -> impl Fn(i32) + Send + Sync + 'static {
    let peer = ctx.peer.clone();
    let token = progress_token(ctx);
    move |p_percent| {
        let Some(tok) = token.clone() else { return };
        let peer = peer.clone();
        let progress = 0.20 + (p_percent as f64 / 100.0) * 0.78;
        tokio::spawn(async move {
            let _ = peer
                .notify_progress(
                    ProgressNotificationParam::new(tok, progress)
                        .with_total(1.0)
                        .with_message(format!("whisper {p_percent}%")),
                )
                .await;
        });
    }
}

fn progress_token(ctx: &RequestContext<RoleServer>) -> Option<rmcp::model::ProgressToken> {
    ctx.meta.get_progress_token()
}

async fn emit_progress(ctx: &RequestContext<RoleServer>, progress: f64, message: Option<String>) {
    if let Some(token) = progress_token(ctx) {
        let mut param = ProgressNotificationParam::new(token, progress).with_total(1.0);
        if let Some(m) = message {
            param = param.with_message(m);
        }
        let _ = ctx.peer.notify_progress(param).await;
    }
}

#[tool_handler]
impl ServerHandler for WhisperServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.instructions = Some(
            "Whisper large-v3 + ffmpeg video analysis on a dedicated GPU. \
             DEFAULT to the light path: `transcribe_file` (local, path under DATA_DIR) \
             or `transcribe_youtube` (URL) return the transcript plus basic info \
             (duration, resolution, language) and NO images — cheap. Pass \
             `timestamps:true` for `[MM:SS]` lines that anchor frame windows. \
             Only when you actually need to SEE the video, load images: `extract_frames` \
             (page one [start_secs, end_secs] window; near-duplicate frames are dropped \
             automatically via `dedup`) or `analyze_file`/`analyze_youtube` (transcript + \
             frames in one shot — best for a SHORT clip). `estimate_video_cost` previews \
             token cost. Frames are heavy (~1.2k tokens each at medium and they accumulate), \
             so keep windows lean and prefer scene sampling. For a WHOLE long video, \
             transcribe once then page successive windows (timestamps stay absolute) rather \
             than one huge call — a disposable subagent can do the paging in its own context \
             and report back text. Progress notifications fire if the client sends a progressToken."
                .into(),
        );
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_span_full_when_no_window() {
        assert_eq!(window_span(None, None, 600.0), 600.0);
        assert_eq!(window_span(Some(0.0), None, 600.0), 600.0);
    }

    #[test]
    fn window_span_uses_bounds() {
        // Explicit [start, end] → end - start (drives the fallback fps over the
        // window, not the whole 600s video).
        assert_eq!(window_span(Some(120.0), Some(180.0), 600.0), 60.0);
        // Open-ended window → from start to full duration.
        assert_eq!(window_span(Some(120.0), None, 600.0), 480.0);
    }

    #[test]
    fn window_span_ignores_bogus_bounds() {
        // end <= start or non-finite → treat as open-ended.
        assert_eq!(window_span(Some(200.0), Some(100.0), 600.0), 400.0);
        assert_eq!(window_span(None, Some(f64::NAN), 600.0), 600.0);
    }

    #[test]
    fn clamp_max_frames_respects_raised_cap() {
        assert_eq!(clamp_max_frames(0), 1);
        assert_eq!(clamp_max_frames(10), 10);
        assert_eq!(clamp_max_frames(60), 60);
        assert_eq!(clamp_max_frames(1000), MAX_FRAMES_CAP);
    }

    #[test]
    fn clamp_uniform_fps_bounds_rate_and_total() {
        // Absolute rate cap (short window): 600/120 = 5.0 == MAX_UNIFORM_FPS.
        assert_eq!(clamp_uniform_fps(100_000.0, 120.0), 5.0);
        // Span cap binds on a long window: 600 frames / 600 s = 1.0 fps ceiling.
        assert_eq!(clamp_uniform_fps(100_000.0, 600.0), 1.0);
        // A sane request passes through untouched.
        assert_eq!(clamp_uniform_fps(0.5, 300.0), 0.5);
        // Never collapses to zero, even for an absurd span.
        assert!(clamp_uniform_fps(100_000.0, 1_000_000.0) >= 0.01);
        // Non-finite span falls back to the absolute cap.
        assert_eq!(clamp_uniform_fps(100_000.0, f64::NAN), 5.0);
    }
}
