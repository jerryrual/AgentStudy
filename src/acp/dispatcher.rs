//! Ordered admission of internal turn events as ACP session notifications.

use agent_client_protocol::v1::{
    schema::{SessionId, SessionNotification},
    ConnectionTo,
};
use agent_client_protocol::{Client, Result as AcpResult};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::run::turn_events::TurnEvent;

use super::event_mapping::session_update;

/// Drain one turn's events and wrap every mappable update with one session ID.
///
/// The first synchronous send error stops admission of later updates.
async fn dispatch_with<E>(
    session_id: SessionId,
    context_window: Option<u32>,
    supports_notices: bool,
    mut events: UnboundedReceiver<TurnEvent>,
    mut send: impl FnMut(SessionNotification) -> Result<(), E>,
) -> Result<(), E> {
    while let Some(event) = events.recv().await {
        if let Some(update) = session_update(event, context_window, supports_notices) {
            send(SessionNotification::new(session_id.clone(), update))?;
        }
    }
    Ok(())
}

/// Admit one turn's notifications to an established stable ACP client connection.
///
/// Successful admission does not guarantee transport delivery.
#[allow(dead_code, reason = "consumed by the upcoming ACP prompt handler")]
pub(crate) async fn dispatch_to_client(
    session_id: SessionId,
    context_window: Option<u32>,
    supports_notices: bool,
    events: UnboundedReceiver<TurnEvent>,
    connection: ConnectionTo<Client>,
) -> AcpResult<()> {
    dispatch_with(
        session_id,
        context_window,
        supports_notices,
        events,
        |notification| connection.send_notification(notification),
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::model::{ToolCall, Usage};
    use crate::run::{turn_events, TurnObserver};

    #[tokio::test]
    async fn wraps_mapped_events_with_one_session_and_preserves_order() {
        let (observer, receiver) = turn_events::channel();
        let call = ToolCall {
            id: "call-4".to_string(),
            name: "read_file".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        };
        observer.text_delta("answer");
        observer.usage(&Usage::default());
        observer.tool_started(&call, "src/lib.rs");
        observer.tool_finished(
            &call,
            true,
            Duration::from_millis(3),
            "src/lib.rs",
            "contents",
        );
        drop(observer);

        let mut notifications = Vec::new();
        dispatch_with(
            SessionId::new("session-7"),
            Some(128_000),
            false,
            receiver,
            |notification| {
                notifications.push(notification);
                Ok::<_, ()>(())
            },
        )
        .await
        .unwrap();

        assert_eq!(notifications.len(), 4);
        let encoded: Vec<_> = notifications
            .into_iter()
            .map(|notification| serde_json::to_value(notification).unwrap())
            .collect();
        assert!(encoded
            .iter()
            .all(|value| value["sessionId"] == "session-7"));
        assert_eq!(encoded[0]["update"]["sessionUpdate"], "agent_message_chunk");
        assert_eq!(encoded[1]["update"]["sessionUpdate"], "usage_update");
        assert_eq!(encoded[1]["update"]["used"], 0);
        assert_eq!(encoded[1]["update"]["size"], 128_000);
        assert_eq!(encoded[2]["update"]["sessionUpdate"], "tool_call");
        assert_eq!(encoded[2]["update"]["toolCallId"], "call-4");
        assert_eq!(encoded[3]["update"]["sessionUpdate"], "tool_call_update");
        assert_eq!(encoded[3]["update"]["toolCallId"], "call-4");
    }

    #[tokio::test]
    async fn omits_events_that_require_session_capabilities_or_context() {
        let (observer, receiver) = turn_events::channel();
        observer.usage(&Usage::default());
        observer.info("working");
        observer.error("failed");
        observer.warning("retrying");
        drop(observer);
        let mut attempts = 0;

        dispatch_with(SessionId::new("session-8"), None, false, receiver, |_| {
            attempts += 1;
            Ok::<_, ()>(())
        })
        .await
        .unwrap();

        assert_eq!(attempts, 0);
    }

    #[tokio::test]
    async fn stops_at_the_first_send_failure() {
        let (observer, receiver) = turn_events::channel();
        observer.text_delta("first");
        observer.text_delta("second");
        observer.text_delta("third");
        drop(observer);
        let mut attempts = 0;

        let result = dispatch_with(SessionId::new("session-9"), None, false, receiver, |_| {
            attempts += 1;
            if attempts == 2 {
                Err("closed")
            } else {
                Ok(())
            }
        })
        .await;

        assert_eq!(result, Err("closed"));
        assert_eq!(attempts, 2);
    }

    #[tokio::test]
    async fn closed_empty_channel_completes_without_sending() {
        let (observer, receiver) = turn_events::channel();
        drop(observer);
        let mut sent = false;

        tokio::time::timeout(
            Duration::from_secs(1),
            dispatch_with(
                SessionId::new("session-empty"),
                None,
                false,
                receiver,
                |_| {
                    sent = true;
                    Ok::<_, ()>(())
                },
            ),
        )
        .await
        .expect("dispatcher did not observe channel closure")
        .unwrap();

        assert!(!sent);
    }
}
