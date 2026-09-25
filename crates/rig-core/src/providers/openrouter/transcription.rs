//! OpenRouter's transcription model identifiers.
//!
//! OpenRouter's speech-to-text route takes a JSON body carrying the audio as
//! a base64 `input_audio` block rather than a multipart upload. The wire's
//! transcription decoder reads the reply; the gateway's own usage fields
//! (audio seconds, cost) stay readable on
//! [`TranscriptionResponse::raw`](crate::transcription::TranscriptionResponse::raw).
//!
//! OpenRouter-specific STT options are passed through
//! [`TranscriptionRequestBuilder::additional_params`], which merges into the
//! top level of the JSON body. `response_format` selects `"json"` (default)
//! or `"verbose_json"`; `timestamp_granularities` asks for `"segment"` and/or
//! `"word"` timestamps (only meaningful with `verbose_json`, and silently
//! ignored without it). The `provider` object forwards options to the backing
//! speech provider, keyed by provider slug; that is the only route to
//! provider-native features such as diarization, and it is unavailable on
//! multipart requests, which this route does not use. Some models reject
//! `verbose_json` with a 400 response (for example `openai/gpt-4o-transcribe`);
//! diarization options are best-effort and produce segments without speaker
//! labels when the backing model cannot diarize. A `verbose_json`
//! response decodes into
//! [`TranscriptionResponse`](crate::providers::openai::transcription::TranscriptionResponse),
//! whose segments and words carry optional speaker labels; the normalized
//! response's [`decode_as`](crate::transcription::TranscriptionResponse::decode_as)
//! recovers that typed payload from `raw`.
//!
//! ```no_run
//! use rig_core::driver::{Bind, HasTranscription as _, Socket};
//! use rig_core::providers::openai::wire::{OPENROUTER, OpenAI};
//! use rig_core::providers::openrouter;
//! use rig_core::transcription::TranscriptionRequestBuilder;
//! use serde_json::json;
//!
//! # async fn example(path: &str, http: impl Socket) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env_with(&OPENROUTER)?
//!     .transcription(openrouter::WHISPER_1)
//!     .bind(http);
//! let response = TranscriptionRequestBuilder::from_file(model, path)?
//!     .additional_params(json!({
//!         "response_format": "verbose_json",
//!         "timestamp_granularities": ["segment", "word"],
//!         "provider": {"options": {"azure": {"diarization": {"enabled": true}}}}
//!     }))
//!     .send()
//!     .await?;
//! let verbose: rig_core::providers::openai::transcription::TranscriptionResponse =
//!     response.decode_as()?;
//! # let _ = verbose.segments;
//! # Ok(())
//! # }
//! ```

/// The `openai/whisper-1` model.
pub const WHISPER_1: &str = "openai/whisper-1";
/// The `openai/whisper-large-v3-turbo` model.
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
/// The `openai/whisper-large-v3` model.
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// The `openai/gpt-4o-transcribe` model.
pub const GPT_4O_TRANSCRIBE: &str = "openai/gpt-4o-transcribe";
/// The `openai/gpt-4o-mini-transcribe` model.
pub const GPT_4O_MINI_TRANSCRIBE: &str = "openai/gpt-4o-mini-transcribe";
/// The `google/chirp-3` model.
pub const CHIRP_3: &str = "google/chirp-3";
