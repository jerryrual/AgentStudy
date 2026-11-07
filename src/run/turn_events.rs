//! Ordered bridge from synchronous turn callbacks to async consumers.

use std::time::Duration;

use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::model::{ToolCall, Usage};

use super::TurnObserver;

/// An owned presentation event emitted while a headless agent turn runs.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol adapter")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TurnEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolStarted {
        call: ToolCall,
        summary: String,
    },
    ToolFinished {
        call: ToolCall,
        ok: bool,
        duration: Duration,
        summary: String,
        output: String,
    },
    Usage(Usage),
    Info(String),
    Error(String),
    Warning(String),
}

/// Nonblocking synchronous observer backed by an ordered async channel.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol adapter")
)]
pub(crate) struct ChannelTurnObserver {
    sender: UnboundedSender<TurnEvent>,
}

/// Create an observer and the receiver drained by an async dispatcher.
///
/// The channel is intentionally unbounded: observer callbacks cannot await and
/// dropping protocol events under backpressure would corrupt the remote turn
/// view. The consumer must drain it for the lifetime of the turn.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol adapter")
)]
pub(crate) fn channel() -> (ChannelTurnObserver, UnboundedReceiver<TurnEvent>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    (ChannelTurnObserver { sender }, receiver)
}

#[cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP protocol adapter")
)]
impl ChannelTurnObserver {
    fn send(&self, event: TurnEvent) {
        let _ = self.sender.send(event);
    }
}

impl TurnObserver for ChannelTurnObserver {
    fn text_delta(&self, delta: &str) {
        self.send(TurnEvent::TextDelta(delta.to_string()));
    }

    fn reasoning_delta(&self, delta: &str) {
        self.send(TurnEvent::ReasoningDelta(delta.to_string()));
    }

    fn tool_started(&self, call: &ToolCall, summary: &str) {
        self.send(TurnEvent::ToolStarted {
            call: call.clone(),
            summary: summary.to_string(),
        });
    }

    fn tool_finished(
        &self,
        call: &ToolCall,
        ok: bool,
        duration: Duration,
        summary: &str,
        output: &str,
    ) {
        self.send(TurnEvent::ToolFinished {
            call: call.clone(),
            ok,
            duration,
            summary: summary.to_string(),
            output: output.to_string(),
        });
    }

    fn usage(&self, usage: &Usage) {
        self.send(TurnEvent::Usage(*usage));
    }

    fn info(&self, message: &str) {
        self.send(TurnEvent::Info(message.to_string()));
    }

    fn error(&self, message: &str) {
        self.send(TurnEvent::Error(message.to_string()));
    }

    fn warning(&self, message: &str) {
        self.send(TurnEvent::Warning(message.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn observer_forwards_owned_events_in_callback_order() {
        let (observer, mut receiver) = channel();
        let call = ToolCall {
            id: "call-11".to_string(),
            name: "read_file".to_string(),
            arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
        };
        let usage = Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
        };

        observer.text_delta("answer");
        observer.reasoning_delta("thought");
        observer.tool_started(&call, "src/lib.rs");
        observer.tool_finished(
            &call,
            true,
            Duration::from_millis(12),
            "src/lib.rs",
            "contents",
        );
        observer.usage(&usage);
        observer.info("working");
        observer.error("failed");
        observer.warning("retrying");
        drop(observer);

        let mut events = Vec::new();
        while let Some(event) = receiver.recv().await {
            events.push(event);
        }

        assert_eq!(
            events,
            vec![
                TurnEvent::TextDelta("answer".to_string()),
                TurnEvent::ReasoningDelta("thought".to_string()),
                TurnEvent::ToolStarted {
                    call: call.clone(),
                    summary: "src/lib.rs".to_string(),
                },
                TurnEvent::ToolFinished {
                    call,
                    ok: true,
                    duration: Duration::from_millis(12),
                    summary: "src/lib.rs".to_string(),
                    output: "contents".to_string(),
                },
                TurnEvent::Usage(usage),
                TurnEvent::Info("working".to_string()),
                TurnEvent::Error("failed".to_string()),
                TurnEvent::Warning("retrying".to_string()),
            ]
        );
    }

    #[test]
    fn closed_receiver_does_not_block_or_panic_callbacks() {
        let (observer, receiver) = channel();
        drop(receiver);

        observer.text_delta("ignored");
        observer.error("also ignored");
    }
}
