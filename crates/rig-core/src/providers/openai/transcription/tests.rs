use super::*;

/// The two live shapes, and the catch-all that keeps a third from failing
/// the transcription. Recorded turns of the first two are replayed in
/// `transcription_usage_matrix`; this pins the decode itself, including
/// the shapes no live model produces.
#[test]
fn usage_decodes_both_billing_shapes_and_keeps_unknown_ones() {
    fn usage(body: &str) -> Option<TranscriptionUsage> {
        serde_json::from_str::<TranscriptionResponse>(body)
            .expect("response should decode")
            .usage
    }

    assert_eq!(
        usage(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#),
        Some(TranscriptionUsage::Duration {
            r#type: DurationTag::Duration,
            seconds: 6.0
        })
    );
    assert_eq!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "input_token_details":{"audio_tokens":54,"text_tokens":0},
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            r#type: TokensTag::Tokens,
            input_tokens: 54,
            input_token_details: Some(TranscriptionInputTokenDetails {
                audio_tokens: 54,
                text_tokens: 0,
            }),
            output_tokens: 16,
            total_tokens: 70,
        })
    );
    // The breakdown is optional: a provider that omits it still decodes as
    // a token-billed turn rather than falling to the catch-all.
    assert_eq!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            r#type: TokensTag::Tokens,
            input_tokens: 54,
            input_token_details: None,
            output_tokens: 16,
            total_tokens: 70,
        })
    );
    // The tag decides, not which optional keys are present: a token-billed
    // payload that also reported `seconds` must not decode as a duration
    // and drop every token count.
    assert!(matches!(
        usage(
            r#"{"text":"hi","usage":{"type":"tokens","seconds":6,"input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
        ),
        Some(TranscriptionUsage::Tokens {
            total_tokens: 70,
            ..
        })
    ));
    assert!(matches!(
        usage(r#"{"text":"hi","usage":{"type":"credits","spent":3}}"#),
        Some(TranscriptionUsage::Other(_))
    ));
    // A token-shaped payload missing a required total degrades to the
    // catch-all rather than failing the transcription.
    assert!(matches!(
        usage(r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54}}"#),
        Some(TranscriptionUsage::Other(_))
    ));
    assert_eq!(usage(r#"{"text":"hi"}"#), None);
    assert_eq!(usage(r#"{"text":"hi","usage":null}"#), None);
}

/// The duration-billed shape reports no token counts, so normalization must
/// leave the usage empty rather than inventing zeros; the token-billed shape
/// carries its counts across.
#[test]
fn normalize_carries_token_billing_and_leaves_duration_billing_empty() {
    fn normalized(body: &str) -> crate::completion::Usage {
        serde_json::from_str::<TranscriptionResponse>(body)
            .expect("response should decode")
            .normalize("openai")
            .expect("normalization is infallible for this wire")
            .usage
    }

    let tokens = normalized(
        r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
               "output_tokens":16,"total_tokens":70}}"#,
    );
    assert_eq!(tokens.input_tokens, Some(54));
    assert_eq!(tokens.output_tokens, Some(16));
    assert_eq!(tokens.total_tokens, Some(70));

    let duration = normalized(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#);
    assert_eq!(duration.input_tokens, None);
    assert_eq!(duration.output_tokens, None);
    assert_eq!(duration.total_tokens, None);
}

/// The plain `json` response format decodes with every verbose field unset.
#[test]
fn plain_json_response_leaves_verbose_fields_unset() {
    let response: TranscriptionResponse =
        serde_json::from_str(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#)
            .expect("response should decode");

    assert_eq!(response.text, "hi");
    assert!(response.language.is_none());
    assert!(response.duration.is_none());
    assert!(response.task.is_none());
    assert!(response.confidence.is_none());
    assert!(response.segments.is_none());
    assert!(response.words.is_none());
}

/// A `verbose_json` response with speaker labels, mirroring OpenRouter's
/// diarized shape, decodes the segments and words with their speakers.
#[test]
fn verbose_json_response_decodes_segments_words_and_speakers() {
    let response: TranscriptionResponse = serde_json::from_str(
        r#"{"text":"Hello there. Hi, how are you?",
            "language":"en","duration":6.4,"task":"transcribe","confidence":0.97,
            "segments":[
                {"id":0,"seek":0,"start":0.0,"end":1.2,"text":"Hello there.",
                 "tokens":[1,2],"temperature":0.0,"avg_logprob":-0.3,
                 "compression_ratio":1.4,"no_speech_prob":0.01,"speaker":0}
            ],
            "words":[
                {"word":"Hello","start":0.0,"end":0.4,"speaker":0,"confidence":0.99}
            ]}"#,
    )
    .expect("response should decode");

    assert_eq!(response.language.as_deref(), Some("en"));
    assert_eq!(response.duration, Some(6.4));
    assert_eq!(response.task.as_deref(), Some("transcribe"));
    assert_eq!(response.confidence, Some(0.97));

    let segments = response.segments.expect("segments should decode");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].id, Some(0));
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 1.2);
    assert_eq!(segments[0].text, "Hello there.");
    assert_eq!(segments[0].speaker, Some(0));

    let words = response.words.expect("words should decode");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].word, "Hello");
    assert_eq!(words[0].speaker, Some(0));
    assert_eq!(words[0].confidence, Some(0.99));
}

/// Provider extras inside segments are Whisper diagnostics that only some
/// backends send; they must be dropped silently rather than failing the
/// transcription.
#[test]
fn verbose_json_response_ignores_unmodeled_segment_extras() {
    let response: TranscriptionResponse = serde_json::from_str(
        r#"{"text":"Hello there.",
            "segments":[
                {"id":0,"start":0.0,"end":1.2,"text":"Hello there.",
                 "avg_logprob":-0.3,"no_speech_prob":0.01,"future_field":"?"}
            ]}"#,
    )
    .expect("response should decode");

    let segments = response.segments.expect("segments should decode");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].speaker, None);
}

/// Verbose payloads without speaker labels (non-diarizing backends) decode
/// the same fields with speakers unset.
#[test]
fn verbose_json_response_without_speakers_decodes() {
    let response: TranscriptionResponse = serde_json::from_str(
        r#"{"text":"Hello there.",
            "segments":[{"id":0,"start":0.0,"end":1.2,"text":"Hello there."}],
            "words":[{"word":"Hello","start":0.0,"end":0.4}]}"#,
    )
    .expect("response should decode");

    let segments = response.segments.expect("segments should decode");
    assert_eq!(segments[0].speaker, None);
    let words = response.words.expect("words should decode");
    assert_eq!(words[0].speaker, None);
    assert_eq!(words[0].confidence, None);
}
