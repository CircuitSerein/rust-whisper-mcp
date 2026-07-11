use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

/// One kept transcript segment with its start time. Used to render the
/// `[MM:SS] text` transcript and to align frames to speech by timestamp.
#[derive(Debug, Clone)]
pub struct TimedSegment {
    /// Segment start time in centiseconds (whisper-rs `start_timestamp`).
    pub start_cs: i64,
    pub text: String,
}

#[derive(Debug)]
pub struct TranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub segments: usize,
    /// Per-segment start + text, after the same dedup backstop as `text`.
    pub timed: Vec<TimedSegment>,
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
        params.set_suppress_nst(true);
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
        let n = state.full_n_segments();
        // Collect (start_cs, trimmed non-empty text) per segment in order, then
        // apply the anti-repetition backstop in one pure pass. Doing it inside
        // `dedup_segments` keeps the concatenated `text` and the per-segment
        // `timed` list (used for [MM:SS] frame alignment) perfectly in lockstep.
        let mut raw: Vec<(i64, String)> = Vec::with_capacity(n as usize);
        for i in 0..n {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };
            if let Ok(seg) = segment.to_str_lossy() {
                let seg = seg.trim();
                if seg.is_empty() {
                    continue;
                }
                raw.push((segment.start_timestamp(), seg.to_string()));
            }
        }
        let (text, timed) = dedup_segments(&raw);

        // whisper-rs 0.16: full_lang_id_from_state returns a bare i32; get_lang_str
        // still returns Option<&'static str> (None for an unset/invalid id), which
        // preserves the original "None when no language detected" behaviour.
        let language = get_lang_str(state.full_lang_id_from_state()).map(|s| s.to_string());

        Ok(TranscriptionOutput {
            text,
            language,
            segments: n as usize,
            timed,
        })
    }
}

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

/// Anti-repetition backstop: from ordered (start_cs, non-empty text) segments,
/// build the space-joined transcript and the matching per-segment list, dropping
/// runs of the same phrase past `MAX_REPEAT` consecutive copies. A short repeat
/// (legit "Да. Да.") is kept; a runaway loop (the same sentence ×1180) is capped.
/// `text` here is byte-for-byte what the previous inline loop produced.
fn dedup_segments(raw: &[(i64, String)]) -> (String, Vec<TimedSegment>) {
    const MAX_REPEAT: u32 = 3;
    let mut text = String::new();
    let mut timed = Vec::with_capacity(raw.len());
    let mut prev: &str = "";
    let mut repeat = 0u32;
    for (start_cs, seg) in raw {
        let seg = seg.as_str();
        if seg == prev {
            repeat += 1;
            if repeat >= MAX_REPEAT {
                continue;
            }
        } else {
            repeat = 0;
            prev = seg;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(seg);
        timed.push(TimedSegment {
            start_cs: *start_cs,
            text: seg.to_string(),
        });
    }
    (text, timed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_keeps_at_most_three_consecutive_repeats() {
        let raw: Vec<(i64, String)> = vec![
            (0, "Да.".into()),
            (100, "Да.".into()),
            (200, "Да.".into()),
            (300, "Да.".into()),  // 4th identical — dropped
            (400, "Да.".into()),  // 5th — dropped
            (500, "Нет.".into()), // distinct — kept, resets the run
        ];
        let (text, timed) = dedup_segments(&raw);
        assert_eq!(text, "Да. Да. Да. Нет.");
        assert_eq!(timed.len(), 4);
        assert_eq!(timed[0].start_cs, 0);
        assert_eq!(timed[3].start_cs, 500);
        assert_eq!(timed[3].text, "Нет.");
    }

    #[test]
    fn dedup_retains_all_distinct_segments() {
        let raw: Vec<(i64, String)> =
            vec![(0, "one".into()), (50, "two".into()), (90, "three".into())];
        let (text, timed) = dedup_segments(&raw);
        assert_eq!(text, "one two three");
        assert_eq!(timed.len(), 3);
        assert_eq!(timed[1].start_cs, 50);
    }

    #[test]
    fn dedup_resets_run_after_a_different_segment() {
        // A repeated phrase, broken by another, may repeat again up to the cap.
        let raw: Vec<(i64, String)> = vec![
            (0, "a".into()),
            (10, "a".into()),
            (20, "a".into()),
            (30, "a".into()), // dropped (4th)
            (40, "b".into()), // resets
            (50, "a".into()), // new run — kept
        ];
        let (text, timed) = dedup_segments(&raw);
        assert_eq!(text, "a a a b a");
        assert_eq!(timed.len(), 5);
    }

    #[test]
    fn dedup_empty_input() {
        let (text, timed) = dedup_segments(&[]);
        assert!(text.is_empty());
        assert!(timed.is_empty());
    }
}
