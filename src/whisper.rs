use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

#[derive(Debug)]
pub struct TranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub segments: usize,
}

/// Holds a loaded whisper.cpp model. CUDA-backed when built with the `cuda` feature.
/// Inference is serialized through a Mutex — single GPU, single context.
pub struct Engine {
    ctx: Mutex<WhisperContext>,
}

impl Engine {
    pub fn load(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(anyhow!("model file not found: {model_path}"));
        }
        tracing::info!("loading whisper model: {model_path}");
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .context("WhisperContext::new")?;
        tracing::info!("whisper model loaded");
        Ok(Self {
            ctx: Mutex::new(ctx),
        })
    }

    /// Transcribe a 16kHz mono f32 PCM buffer.
    /// Returns concatenated segment text plus detected language and segment count.
    pub fn transcribe_pcm(
        &self,
        pcm: &[f32],
        language: Option<&str>,
        on_progress: impl Fn(i32) + Send + Sync + 'static,
    ) -> Result<TranscriptionOutput> {
        let ctx = self.ctx.lock().map_err(|_| anyhow!("whisper mutex poisoned"))?;
        let mut state = ctx.create_state().context("create whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(num_cpus());
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Reduce hallucinated repetition on silence/outro music.
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_single_segment(false);
        // Explicit "auto" triggers whisper.cpp's language detection.
        // Default (unset) is "en", which silently mis-transcribes non-English.
        params.set_language(Some(language.unwrap_or("auto")));
        params.set_progress_callback_safe(move |p| on_progress(p));

        state.full(params, pcm).context("whisper full")?;

        let n = state.full_n_segments().context("segment count")?;
        let mut text = String::new();
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(seg.trim());
            }
        }

        let language = state
            .full_lang_id_from_state()
            .ok()
            .and_then(get_lang_str)
            .map(|s| s.to_string());

        Ok(TranscriptionOutput {
            text,
            language,
            segments: n as usize,
        })
    }
}

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}
