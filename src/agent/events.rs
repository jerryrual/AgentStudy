//! Events emitted by the agent while a turn runs.
//!
//! The agent never draws anything itself: the TUI and `icode run` implement
//! [`EventSink`] to render streaming text, tool calls and usage.

use std::time::Duration;

use crate::compaction::{CompressionBatch, ContextProjection};
use crate::model::{Message, Usage};

use super::TurnResult;

/// Receives incremental updates from the agent loop.
pub trait EventSink: Send + Sync {
    /// A new user turn started.
    fn turn_started(&self) {}

    /// A chunk of assistant text arrived.
    fn text_delta(&self, _delta: &str) {}

    /// A chunk of model reasoning arrived (DeepSeek `reasoning_content`).
    fn reasoning_delta(&self, _delta: &str) {}

    /// A tool call started.
    fn tool_started(&self, _name: &str, _summary: &str) {}

    /// A tool call finished; `output` is a shortened preview of its result.
    fn tool_finished(
        &self,
        _name: &str,
        _ok: bool,
        _duration: Duration,
        _summary: &str,
        _output: &str,
    ) {
    }

    /// Persist an atomic compression batch before its projection is sent to the provider.
    ///
    /// Returns `true` when the batch was durably checkpointed. The synchronous
    /// contract intentionally makes persistence complete before the request proceeds.
    fn persist_compression(&self, _batch: &CompressionBatch) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Persist one context projection before its replacement is sent to the provider.
    ///
    /// A `true` result guarantees earlier messages from the same turn were
    /// persisted first and the projection is durably ordered after them.
    fn persist_context_projection(
        &self,
        _source: &[Message],
        _projection: &ContextProjection,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// A message was appended to the conversation (used for persistence).
    fn message_added(&self, _message: &Message) {}

    /// Token usage for one model response.
    fn usage(&self, _usage: &Usage) {}

    /// Informational status line.
    fn info(&self, _message: &str) {}

    /// Recoverable or terminal error line.
    fn error(&self, _message: &str) {}

    /// The turn finished.
    fn turn_finished(&self, _result: &TurnResult) {}
}

/// A sink that discards everything; useful for tests and library callers.
pub struct NullSink;

impl EventSink for NullSink {}
