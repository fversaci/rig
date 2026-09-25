use super::*;
use crate::providers::openai::transcription::TranscriptionResponse as OpenAiTranscriptionResponse;
use crate::{http_client, provider_response};
use http::StatusCode;

/// The raw provider document decodes back into the provider's typed response,
/// so verbose transcription fields are reachable through the generic type.
#[test]
fn decode_as_recovers_the_typed_provider_response() {
    let raw = serde_json::json!({
        "text": "Hello there.",
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello there.", "speaker": 0}
        ],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": 0, "confidence": 0.99}
        ]
    });
    let response = TranscriptionResponse::new("Hello there.", "openrouter").with_raw(raw);

    let verbose: OpenAiTranscriptionResponse = response
        .decode_as()
        .expect("the raw document should decode into the provider type");

    assert_eq!(verbose.text, "Hello there.");
    assert_eq!(verbose.language.as_deref(), Some("en"));
    let segments = verbose.segments.expect("segments should decode");
    assert_eq!(segments[0].speaker, Some(0));
    let words = verbose.words.expect("words should decode");
    assert_eq!(words[0].confidence, Some(0.99));
}

/// A raw document that does not fit the caller's type fails with a
/// deserialization error rather than a provider error.
#[test]
fn decode_as_fails_when_the_shape_does_not_match() {
    let response =
        TranscriptionResponse::new("Hello there.", "openrouter").with_raw(serde_json::json!([]));

    assert!(response.decode_as::<OpenAiTranscriptionResponse>().is_err());
}

#[test]
fn transcription_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"rate limited"}}"#;
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "rate limited" } }))
    );
}

#[test]
fn transcription_error_provider_response_helpers_with_http_non_success() {
    let body = r#"{"error":{"message":"bad request"}}"#;
    let error = ProviderError::from_transport_error(http_client::Error::non_success_with_details(
        StatusCode::BAD_REQUEST,
        http::HeaderMap::new(),
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "bad request" } }))
    );
}

#[test]
fn transcription_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("not json".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("not json"));
    assert!(error.provider_response_json().is_err());
}

#[test]
fn transcription_error_provider_error_is_not_a_provider_response() {
    let error = ProviderError::Provider("internal diagnostic".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}

#[test]
fn transcription_error_provider_response_helpers_with_unrelated_variant() {
    let error = ProviderError::Response("parse failed".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}
