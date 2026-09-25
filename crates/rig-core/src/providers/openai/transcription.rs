use crate::completion::Usage;
use crate::error::ProviderError;
use crate::transcription;
use crate::transcription::NormalizeTranscriptionResponse;
use serde::{Deserialize, Serialize};

pub const WHISPER_1: &str = "whisper-1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    /// Optional endpoint-reported usage. Token counts normalize into
    /// [`transcription::TranscriptionResponse::usage`]; duration remains raw.
    #[serde(default)]
    pub usage: Option<TranscriptionUsage>,
    /// Detected or requested language as a BCP-47 style tag, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Length of the transcribed audio, in seconds, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// The operation the provider performed, typically `"transcribe"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    /// Overall transcript confidence in `[0, 1]`, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Time-aligned transcript segments, when the provider produced them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<TranscriptionSegment>>,
    /// Time-aligned transcript words, when the provider produced them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<TranscriptionWord>>,
}

/// A time-aligned span of the transcript. Provider extras (Whisper
/// diagnostics, for instance) are ignored rather than failing the decode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    /// Segment index within the transcript, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u64>,
    /// Offset from the audio start, in seconds.
    pub start: f64,
    /// Offset from the audio start where this segment ends, in seconds.
    pub end: f64,
    /// The transcribed text of this segment.
    pub text: String,
    /// Zero-based speaker label, when the provider diarized. Numeric labels
    /// cover the diarizing backends on this wire; a backend emitting another
    /// label shape leaves this unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u64>,
}

/// A time-aligned word of the transcript. Provider extras are ignored rather
/// than failing the decode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionWord {
    /// The transcribed word.
    pub word: String,
    /// Offset from the audio start, in seconds.
    pub start: f64,
    /// Offset from the audio start where this word ends, in seconds.
    pub end: f64,
    /// Zero-based speaker label, when the provider diarized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u64>,
    /// Word confidence in `[0, 1]`, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Duration or token accounting selected by the wire's `type` tag.
/// Payloads that do not match a modeled shape are preserved verbatim in `Other`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TranscriptionUsage {
    /// Duration-billed models.
    Duration {
        /// Always `"duration"`; pins this variant to its wire tag.
        r#type: DurationTag,
        /// Length of the audio, in seconds.
        seconds: f64,
    },
    /// Token-billed models.
    Tokens {
        /// Always `"tokens"`; pins this variant to its wire tag.
        r#type: TokensTag,
        /// Tokens consumed by the audio and any prompt.
        input_tokens: u64,
        /// How the input tokens split between audio and text, when the
        /// provider breaks it down. The two are billed at different rates, so
        /// `input_tokens` alone does not determine what a turn cost.
        #[serde(default)]
        input_token_details: Option<TranscriptionInputTokenDetails>,
        /// Tokens in the transcript.
        output_tokens: u64,
        /// `input_tokens + output_tokens`, as the provider reported it.
        total_tokens: u64,
    },
    /// A shape this version does not model, preserved as sent.
    Other(serde_json::Value),
}

/// The wire tag of [`TranscriptionUsage::Duration`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DurationTag {
    /// `"duration"`.
    Duration,
}

/// The wire tag of [`TranscriptionUsage::Tokens`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokensTag {
    /// `"tokens"`.
    Tokens,
}

/// How a token-billed transcription's input tokens split by modality.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionInputTokenDetails {
    /// Input tokens attributable to the audio.
    #[serde(default)]
    pub audio_tokens: u64,
    /// Input tokens attributable to text (a prompt, for instance).
    #[serde(default)]
    pub text_tokens: u64,
}

impl NormalizeTranscriptionResponse for TranscriptionResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<transcription::TranscriptionResponse, ProviderError> {
        let usage = match &self.usage {
            Some(TranscriptionUsage::Tokens {
                input_tokens,
                output_tokens,
                total_tokens,
                ..
            }) => Usage {
                input_tokens: Some(*input_tokens),
                output_tokens: Some(*output_tokens),
                total_tokens: Some(*total_tokens),
                ..Default::default()
            },
            // Duration billing reports no token counts; the seconds stay
            // reachable on the raw payload.
            Some(TranscriptionUsage::Duration { .. })
            | Some(TranscriptionUsage::Other(_))
            | None => Usage::default(),
        };
        Ok(transcription::TranscriptionResponse::new(self.text, provider).with_usage(usage))
    }
}

#[cfg(test)]
mod tests;
