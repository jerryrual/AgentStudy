mod common;

use icode::compaction;
use icode::model::{Message, ModelClient};

use common::{mock_profile, MockResponse, MockServer, RecordingSink};

#[tokio::test]
async fn model_compaction_sends_transcript_as_data_and_returns_summary() {
    let server = MockServer::new(|path, body| {
        assert_eq!(path, "/v1/chat/completions");
        assert!(body.get("tools").is_none());
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("untrusted data"));
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("fix the parser"));
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"content":"Parser fixed; "}}]}"#,
            r#"{"choices":[{"delta":{"content":"tests pass."},"finish_reason":"stop"}],"usage":{"prompt_tokens":40,"completion_tokens":6,"total_tokens":46}}"#,
        ])
    });
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();

    let result = compaction::compact(
        &client,
        &[
            Message::user("fix the parser"),
            Message::assistant(icode::model::AssistantMessage::text("done")),
        ],
    )
    .await
    .unwrap();

    assert_eq!(result.summary, "Parser fixed; tests pass.");
    assert_eq!(result.usage.total_tokens, 46);
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn model_compaction_trims_old_history_to_recover_from_context_exhaustion() {
    let server = MockServer::new(|_, body| {
        let excerpt = body["messages"][1]["content"].as_str().unwrap();
        assert!(excerpt.contains("Earlier transcript entries omitted"));
        assert!(excerpt.contains("recent decision"));
        assert!(!excerpt.contains("obsolete details"));
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"content":"Recent decision retained."},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":4,"total_tokens":104}}"#,
        ])
    });
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(1_200);
    profile.max_tokens = Some(128);
    let client = ModelClient::from_profile(&profile).unwrap();
    let messages = [
        Message::user(format!("obsolete details {}", "x".repeat(2_000))),
        Message::user("recent decision"),
    ];

    let ordinary_error = client
        .complete(&messages, &[], &RecordingSink::default())
        .await
        .unwrap_err();
    assert!(ordinary_error.to_string().contains("exhausts"));
    let result = compaction::compact(&client, &messages).await.unwrap();

    assert_eq!(result.summary, "Recent decision retained.");
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn model_compaction_truncates_one_oversized_entry_on_utf8_boundaries() {
    let server = MockServer::new(|_, body| {
        let excerpt = body["messages"][1]["content"].as_str().unwrap();
        assert!(excerpt.contains("Earlier transcript entries omitted"));
        assert!(excerpt.contains("最终决定"));
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"content":"Final decision retained."},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":4,"total_tokens":104}}"#,
        ])
    });
    let mut profile = mock_profile(server.base_url());
    profile.context_window = Some(1_200);
    profile.max_tokens = Some(128);
    let client = ModelClient::from_profile(&profile).unwrap();
    let message = Message::user(format!("{}最终决定", "历史内容".repeat(500)));

    let result = compaction::compact(&client, &[message]).await.unwrap();

    assert_eq!(result.summary, "Final decision retained.");
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn model_compaction_rejects_missing_finish_reason() {
    let server = MockServer::new(|_, _| {
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"content":"Apparently complete summary"}}],"usage":{"prompt_tokens":40,"completion_tokens":6,"total_tokens":46}}"#,
        ])
    });
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();

    let error = compaction::compact(&client, &[Message::user("important context")])
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("finish reason: missing"),
        "{error:#}"
    );
}

#[tokio::test]
async fn model_compaction_rejects_truncated_summaries() {
    let server = MockServer::new(|_, _| {
        MockResponse::sse(&[
            r#"{"choices":[{"delta":{"content":"Incomplete summary"},"finish_reason":"length"}],"usage":{"prompt_tokens":40,"completion_tokens":6,"total_tokens":46}}"#,
        ])
    });
    let client = ModelClient::from_profile(&mock_profile(server.base_url())).unwrap();

    let error = compaction::compact(&client, &[Message::user("important context")])
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("finish reason: length"),
        "{error:#}"
    );
}
