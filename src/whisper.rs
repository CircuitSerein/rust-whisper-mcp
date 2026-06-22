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
        let ctx = self
            .ctx
            .lock()
            .map_err(|_| anyhow!("whisper mutex poisoned"))?;
        let mut state = ctx.create_state().context("create whisper state")?;

        // best_of > 1 gives the temperature fallback real candidates to pick
        // from when a decode is rejected as repetitive/low-confidence.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_n_threads(num_cpus());
        params.set_translate(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // --- Anti-hallucination / anti-repetition ---
        // Degraded-mic or near-silent audio makes whisper.cpp loop the same
        // phrase. Layered defences:
        //   1. no_context: don't feed prior text back in (no cross-window carry).
        //   2. temperature fallback (temperature_inc): retry hotter when a decode
        //      is repetitive (low entropy) or low-confidence (low logprob).
        //   3. suppress_nst: drop non-speech tokens.
        //   4. entropy_thold raised 2.4 -> 2.8: reject repetitive decodes harder.
        // NOTE: no_speech_thold is a no-op in whisper.cpp (unimplemented), so
        // silence detection can't be relied on — hence the segment-dedup guard
        // below as the hard backstop.
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);
        params.set_single_segment(false);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.8);
        params.set_logprob_thold(-1.0);
        // Explicit "auto" triggers whisper.cpp's language detection.
        // Default (unset) is "en", which silently mis-transcribes non-English.
        params.set_language(Some(language.unwrap_or("auto")));
        params.set_progress_callback_safe(on_progress);

        state.full(params, pcm).context("whisper full")?;

        // Hard backstop against runaway repetition: even if whisper still loops
        // on unrecoverable audio, cap consecutive identical segments so the
        // output can never again become the same sentence x1180. A short repeat
        // is kept (legit "Да. Да."); the rest is dropped.
        let n = state.full_n_segments().context("segment count")?;
        const MAX_REPEAT: u32 = 3;
        let mut text = String::new();
        let mut prev = String::new();
        let mut repeat = 0u32;
        for i in 0..n {
            if let Ok(seg) = state.full_get_segment_text(i) {
                let seg = seg.trim();
                if seg.is_empty() {
                    continue;
                }
                if seg == prev {
                    repeat += 1;
                    if repeat >= MAX_REPEAT {
                        continue;
                    }
                } else {
                    repeat = 0;
                    prev = seg.to_string();
                }
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(seg);
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
