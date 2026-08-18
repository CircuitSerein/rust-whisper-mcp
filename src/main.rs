mod handler;
mod media;
mod paths;
mod whisper;

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use serde::Deserialize;
use serde_json::json;
use tracing_subscriber::EnvFilter;

use crate::handler::{
    AnalyzeFileArgs, AnalyzeResult, AnalyzeYoutubeArgs, EstimateArgs, ExtractFramesArgs,
    TokenEstimate, TranscribeResult, WhisperServer,
};

#[derive(Clone)]
struct AppState {
    server: WhisperServer,
}

#[derive(Debug, Deserialize)]
struct ApiFileReq {
    path: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    cleanup: bool,
}

#[derive(Debug, Deserialize)]
struct ApiUploadQuery {
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiYoutubeReq {
    url: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default = "default_true")]
    cleanup: bool,
}

fn default_true() -> bool {
    true
}

/// Host authorities the `/mcp` endpoint accepts, from `MCP_ALLOWED_HOSTS`
/// (comma-separated).
///
/// rmcp 3 validates the inbound `Host` header to block DNS-rebinding attacks on
/// locally running servers, defaulting to loopback only. An entry without a port
/// matches any port, so the default covers `localhost:17800`. Behind a reverse
/// proxy that forwards its own `Host`, list that hostname here; `*` disables the
/// check entirely (only sensible when something in front already validates it).
fn allowed_hosts(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return StreamableHttpServerConfig::default().allowed_hosts;
    };
    if raw == "*" {
        return Vec::new();
    }
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let model_path =
        std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| "/models/ggml-large-v3.bin".into());
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    // Optional blended USD-per-1M-tokens rate; when unset, cost estimates omit `$`.
    let cost_per_mtok = std::env::var("TOKEN_COST_USD_PER_MTOK")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());

    tracing::info!(model = %model_path, data = %data_dir, bind = %bind, "starting rust-whisper-mcp");

    let engine = Arc::new(whisper::Engine::load(&model_path).context("load whisper model")?);
    let server = WhisperServer::new(engine, data_dir.into(), cost_per_mtok);

    // MCP 2026-07-28 is sessionless: rmcp builds a fresh handler per request, so
    // `WhisperServer` must stay cheap to clone and keep its real state behind
    // `Arc` (the model, the heavy-work semaphore). The session manager below is
    // only used by clients that negotiate an older protocol version.
    let mcp_server = server.clone();
    let mcp_config = StreamableHttpServerConfig::default().with_allowed_hosts(allowed_hosts(
        std::env::var("MCP_ALLOWED_HOSTS").ok().as_deref(),
    ));
    tracing::info!(allowed_hosts = ?mcp_config.allowed_hosts, "mcp transport configured");
    let mcp_service = StreamableHttpService::new(
        move || Ok(mcp_server.clone()),
        LocalSessionManager::default().into(),
        mcp_config,
    );

    let state = AppState { server };

    // 4 GiB cap on uploaded bodies so a several-hour video doesn't get rejected.
    const UPLOAD_LIMIT: usize = 4 * 1024 * 1024 * 1024;

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/transcribe/file", post(api_transcribe_file))
        .route("/api/transcribe/youtube", post(api_transcribe_youtube))
        .route(
            "/api/transcribe/upload",
            post(api_transcribe_upload).layer(DefaultBodyLimit::max(UPLOAD_LIMIT)),
        )
        .route("/api/analyze/file", post(api_analyze_file))
        .route("/api/analyze/youtube", post(api_analyze_youtube))
        .route("/api/estimate", post(api_estimate))
        .route("/api/extract-frames", post(api_extract_frames))
        .with_state(state)
        .nest_service("/mcp", mcp_service);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(
        "listening on http://{bind} (MCP at /mcp, REST at /api/transcribe/* and /api/analyze/*)"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn api_transcribe_file(
    State(state): State<AppState>,
    Json(req): Json<ApiFileReq>,
) -> Result<Json<TranscribeResult>, ApiError> {
    let res = state
        .server
        .run_file(&req.path, req.language.as_deref(), req.cleanup, |_| {})
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_transcribe_upload(
    State(state): State<AppState>,
    Query(q): Query<ApiUploadQuery>,
    body: Bytes,
) -> Result<Json<TranscribeResult>, ApiError> {
    let res = state
        .server
        .run_upload(
            body.to_vec(),
            q.filename.as_deref(),
            q.language.as_deref(),
            |_| {},
        )
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_transcribe_youtube(
    State(state): State<AppState>,
    Json(req): Json<ApiYoutubeReq>,
) -> Result<Json<TranscribeResult>, ApiError> {
    let res = state
        .server
        .run_youtube(&req.url, req.language.as_deref(), req.cleanup, |_| {})
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_analyze_file(
    State(state): State<AppState>,
    Json(req): Json<AnalyzeFileArgs>,
) -> Result<Json<AnalyzeResult>, ApiError> {
    let res = state
        .server
        .run_analyze_file(&req, |_| {})
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_analyze_youtube(
    State(state): State<AppState>,
    Json(req): Json<AnalyzeYoutubeArgs>,
) -> Result<Json<AnalyzeResult>, ApiError> {
    let res = state
        .server
        .run_analyze_youtube(&req, |_| {})
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_estimate(
    State(state): State<AppState>,
    Json(req): Json<EstimateArgs>,
) -> Result<Json<TokenEstimate>, ApiError> {
    let res = state
        .server
        .run_estimate(&req)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

async fn api_extract_frames(
    State(state): State<AppState>,
    Json(req): Json<ExtractFramesArgs>,
) -> Result<Json<AnalyzeResult>, ApiError> {
    let res = state
        .server
        .run_extract_frames(&req)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(res))
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        let message = format!("{e:#}");
        // Path-traversal / bad-input errors come from `paths::resolve_in_data_dir`
        // and surface here with "resolve path" context. An unsupported URL scheme is
        // likewise bad client input (from `media::probe_remote`/`download_video`).
        let status = if message.contains("path traversal")
            || message.contains("escapes DATA_DIR")
            || message.contains("outside DATA_DIR")
            || message.contains("unsupported URL scheme")
            || message.contains("invalid window")
            || message.contains("no video stream")
        {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self { status, message }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_hosts_default_to_loopback_only() {
        let hosts = allowed_hosts(None);
        assert!(hosts.contains(&"localhost".to_string()));
        assert!(hosts.contains(&"127.0.0.1".to_string()));
        assert_eq!(hosts, allowed_hosts(Some("   ")));
    }

    #[test]
    fn allowed_hosts_parses_a_comma_separated_list() {
        assert_eq!(
            allowed_hosts(Some("localhost, whisper.internal:8443 ,")),
            vec!["localhost", "whisper.internal:8443"]
        );
    }

    #[test]
    fn allowed_hosts_star_disables_the_check() {
        assert!(allowed_hosts(Some("*")).is_empty());
    }
}
