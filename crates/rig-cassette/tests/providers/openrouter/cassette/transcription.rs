//! Cassette-backed OpenRouter transcription smoke test.

use rig::providers::openrouter;
use rig::transcription::TranscriptionRequestBuilder;

use crate::support::{AUDIO_FIXTURE_PATH, assert_nonempty_response};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn transcription_smoke() {
    with_openrouter_cassette("transcription/transcription_smoke", |client| async move {
        let model = client.transcription(openrouter::WHISPER_1);
        let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
            .expect("should be able to load audio fixture")
            .send()
            .await
            .expect("transcription should succeed");

        assert_nonempty_response(&response.text);
    })
    .await;
}

/// A `verbose_json` request on a non-diarizing model returns time-aligned
/// segments and words without speaker labels.
#[tokio::test]
async fn transcription_verbose_segments_and_words() {
    with_openrouter_cassette(
        "transcription/transcription_verbose_segments_and_words",
        |client| async move {
            let model = client.transcription("nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b");
            let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture")
                .additional_params(serde_json::json!({
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment", "word"]
                }))
                .send()
                .await
                .expect("verbose transcription should succeed");

            let verbose: rig::providers::openai::transcription::TranscriptionResponse = response
                .decode_as()
                .expect("verbose payload should decode into the provider response");

            assert_nonempty_response(&verbose.text);
            let segments = verbose.segments.expect("verbose response should have segments");
            assert!(!segments.is_empty(), "segments should not be empty");
            for segment in &segments {
                assert!(segment.speaker.is_none(), "this model does not diarize");
            }
            let words = verbose.words.expect("verbose response should have words");
            assert!(!words.is_empty(), "words should not be empty");
        },
    )
    .await;
}

/// A `verbose_json` request with Azure diarization options returns segments
/// whose speaker labels identify who spoke each span.
#[tokio::test]
async fn transcription_verbose_diarized() {
    with_openrouter_cassette(
        "transcription/transcription_verbose_diarized",
        |client| async move {
            let model = client.transcription("microsoft/mai-transcribe-2");
            let response = TranscriptionRequestBuilder::from_file(model, AUDIO_FIXTURE_PATH)
                .expect("should be able to load audio fixture")
                .additional_params(serde_json::json!({
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment", "word"],
                    "provider": {"options": {"azure": {"diarization": {"enabled": true}}}}
                }))
                .send()
                .await
                .expect("diarized transcription should succeed");

            let verbose: rig::providers::openai::transcription::TranscriptionResponse = response
                .decode_as()
                .expect("verbose payload should decode into the provider response");

            let segments = verbose.segments.expect("verbose response should have segments");
            assert!(!segments.is_empty(), "segments should not be empty");
            assert!(
                segments.iter().any(|segment| segment.speaker.is_some()),
                "diarized audio should label at least one segment with a speaker"
            );
            let words = verbose.words.expect("verbose response should have words");
            assert!(!words.is_empty(), "words should not be empty");
        },
    )
    .await;
}
