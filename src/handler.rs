use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use rmcp::{
    handler::server::router::tool::ToolRouter,
    model::{CallToolResult, Content, ProgressNotificationParam, ServerCapabilities, ServerInfo},
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

#[tool_router]
impl WhisperServer {
    pub fn new(engine: Arc<Engine>, data_dir: PathBuf) -> Self {
        Self {
            engine,
            data_dir,
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
        Ok(CallToolResult::success(vec![Content::text(res.text)]))
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
        Ok(CallToolResult::success(vec![Content::text(res.text)]))
    }
}

fn short_id() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    id[..8].to_string()
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
                .notify_progress(ProgressNotificationParam {
                    progress_token: tok,
                    progress,
                    total: Some(1.0),
                    message: Some(format!("whisper {p_percent}%")),
                })
                .await;
        });
    }
}

fn progress_token(ctx: &RequestContext<RoleServer>) -> Option<rmcp::model::ProgressToken> {
    ctx.meta.get_progress_token()
}

async fn emit_progress(ctx: &RequestContext<RoleServer>, progress: f64, message: Option<String>) {
    if let Some(token) = progress_token(ctx) {
        let _ = ctx
            .peer
            .notify_progress(ProgressNotificationParam {
                progress_token: token,
                progress,
                total: Some(1.0),
                message,
            })
            .await;
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
