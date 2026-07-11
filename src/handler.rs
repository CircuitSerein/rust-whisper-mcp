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
    /// Delete the source file after successful transcription. Default: false.
    #[serde(default)]
    pub cleanup: bool,
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
}

fn default_true() -> bool {
    true
}

// ============================= video analysis =============================

/// Default frames returned by an analyze call; hard cap the client can't exceed.
/// A modern host resizes each image to ≤1568px anyway and enforces per-message
/// image/byte limits, so we keep the count deliberately small.
const DEFAULT_MAX_FRAMES: u32 = 10;
const MAX_FRAMES_CAP: u32 = 20;
const SCENE_THRESHOLD_DEFAULT: f64 = 0.4;
const SCENE_THRESHOLD_MIN: f64 = 0.1;
const SCENE_THRESHOLD_MAX: f64 = 0.6;
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
    layout: Layout,
    language: Option<String>,
    transcribe: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnalyzeFileArgs {
    /// Path to a video file. Resolved under DATA_DIR, same rules as transcribe_file.
    pub path: String,
    /// Optional ISO-639-1 language code (e.g. "en", "ru"). Auto-detect if omitted.
    #[serde(default)]
    pub language: Option<String>,
    /// Delete the source file after analysis. Default: false.
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
    /// Frames per second for uniform sampling. Default derived from duration.
    #[serde(default)]
    pub fps: Option<f64>,
    /// Max frames to return (hard cap 20). Default 10.
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    /// Result layout: frames (default, one image per frame) or grid (contact sheet).
    #[serde(default)]
    pub layout: Layout,
}

impl AnalyzeFileArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            layout: self.layout,
            language: self.language.clone(),
            transcribe: true,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AnalyzeYoutubeArgs {
    /// YouTube (or any yt-dlp-supported) URL. Video+audio downloaded once.
    pub url: String,
    #[serde(default)]
    pub language: Option<String>,
    /// Delete the downloaded video after analysis. Default: true.
    #[serde(default = "default_true")]
    pub cleanup: bool,
    #[serde(default)]
    pub quality: QualityPreset,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub scene_threshold: Option<f64>,
    #[serde(default)]
    pub fps: Option<f64>,
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default)]
    pub layout: Layout,
}

impl AnalyzeYoutubeArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            layout: self.layout,
            language: self.language.clone(),
            transcribe: true,
        }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExtractFramesArgs {
    /// Path to a video file (resolved under DATA_DIR).
    pub path: String,
    #[serde(default)]
    pub quality: QualityPreset,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub scene_threshold: Option<f64>,
    #[serde(default)]
    pub fps: Option<f64>,
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default)]
    pub layout: Layout,
    /// Delete the source file after extraction. Default: false.
    #[serde(default)]
    pub cleanup: bool,
}

impl ExtractFramesArgs {
    fn opts(&self) -> AnalyzeOpts {
        AnalyzeOpts {
            quality: self.quality,
            sampling: self.sampling,
            scene_threshold: self.scene_threshold,
            fps: self.fps,
            max_frames: clamp_max_frames(self.max_frames),
            layout: self.layout,
            language: None,
            transcribe: false,
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
    #[serde(default)]
    pub quality: QualityPreset,
    /// scene (upper-bound frame count = max_frames) or uniform (duration × fps).
    #[serde(default)]
    pub sampling: Sampling,
    /// Frames per second for uniform sampling. Default derived from duration.
    #[serde(default)]
    pub fps: Option<f64>,
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

        Ok(TranscribeResult {
            text: out.text,
            language: out.language,
            segments: out.segments,
            elapsed_ms: elapsed.as_millis(),
        })
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

        Ok(TranscribeResult {
            text: out.text,
            language: out.language,
            segments: out.segments,
            elapsed_ms: elapsed.as_millis(),
        })
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

        Ok(TranscribeResult {
            text: out.text,
            language: out.language,
            segments: out.segments,
            elapsed_ms: elapsed.as_millis(),
        })
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
        description = "Transcribe a local media file (any format ffmpeg understands) to text using Whisper large-v3 on GPU. Path is resolved under the container's DATA_DIR mount."
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
        Ok(CallToolResult::success(vec![ContentBlock::text(res.text)]))
    }

    #[tool(
        name = "transcribe_youtube",
        description = "Download a YouTube (or any yt-dlp-supported) URL as audio and transcribe to text. The download lands under DATA_DIR and is cleaned up by default."
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
        Ok(CallToolResult::success(vec![ContentBlock::text(res.text)]))
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
        let res = self.analyze_common(&file, args.opts(), on_progress).await;
        if args.cleanup {
            media::delete_quiet(&file).await;
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
        let box_px = opts.quality.box_px();
        let sampling = resolve_sampling(
            opts.sampling,
            opts.scene_threshold,
            opts.fps,
            meta.duration_secs,
            opts.max_frames,
        );
        let params = media::ExtractParams {
            box_px,
            sampling,
            max_frames: opts.max_frames,
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
            let fps = default_uniform_fps(meta.duration_secs, opts.max_frames);
            let uni = media::ExtractParams {
                box_px,
                sampling: media::SamplingMode::Uniform { fps },
                max_frames: opts.max_frames,
            };
            frames = media::extract_keyframes(file, &frames_dir, &uni)
                .await
                .context("uniform fallback extract")?;
            notes.push(format!(
                "[no scene cuts detected — sampled uniformly at {fps:.3} fps]"
            ));
            sampling_used = format!("scene→uniform(fps={fps:.3})");
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
        description = "Estimate the token/cost of analyzing a video (local path or URL) at given settings, WITHOUT running whisper or returning any images. Call this first to preview cost, then analyze_file/analyze_youtube."
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
        description = "Analyze a local video: extract keyframes and transcribe, returning frames as images interleaved with a timestamped transcript so a vision model can 'see' the video. Path resolved under DATA_DIR."
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
        description = "Download a video URL once (video+audio), then extract keyframes and transcribe, returning frames as images plus a timestamped transcript. The download is cleaned up by default."
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
        description = "Extract video keyframes only (no transcription), returning them as images. Same sampling/quality options as analyze_file."
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

/// Assemble the mixed text+image tool result: metadata → frames (per-frame label +
/// image for `frames` layout, or the single grid image) → timestamped transcript.
fn build_content(r: &AnalyzeResult, layout: Layout) -> Vec<ContentBlock> {
    let mut blocks = vec![ContentBlock::text(format_metadata(r))];
    for f in &r.frames {
        if matches!(layout, Layout::Frames) {
            blocks.push(ContentBlock::text(format!(
                "[frame {}] {}",
                f.index, f.mmss
            )));
        }
        blocks.push(ContentBlock::image(f.jpeg_base64.clone(), "image/jpeg"));
    }
    if !r.transcript.is_empty() {
        let body = r
            .transcript
            .iter()
            .map(|l| format!("[{}] {}", l.mmss, l.text))
            .collect::<Vec<_>>()
            .join("\n");
        blocks.push(ContentBlock::text(format!("Transcript:\n{body}")));
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
            "Whisper large-v3 transcription server running on a dedicated GPU. \
             Use `transcribe_file` for local media (path under DATA_DIR) or \
             `transcribe_youtube` for URLs. Progress notifications are sent if \
             the client supplies a progressToken in the request meta."
                .into(),
        );
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}
