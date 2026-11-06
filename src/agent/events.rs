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

    /// The stable previous-turn archive pointer captured when this turn started.
    fn archive_pointer(&self) -> Option<String> {
        None
    }

    /// Render the cumulative current-turn archive manifest for model context.
    fn current_turn_archive_manifest(&self, _file_read_available: bool) -> Option<String> {
        None
    }

    /// Synchronously persist one consumed same-turn user injection.
    ///
    /// A successful return guarantees that the transcript record is complete;
    /// the caller must still pass the strict recovery barrier before provider I/O.
    fn persist_injected_user(&self, _message: &Message) -> anyhow::Result<()> {
        anyhow::bail!("injected user persistence is unavailable")
    }

    /// Durably snapshot the exact recoverable turn before provider work.
    fn persist_recovery_checkpoint(&self, _messages: &[Message]) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Accept a completed-tool snapshot for ordered background persistence.
    /// A later strict barrier or terminal drain surfaces asynchronous failures.
    fn publish_recovery_checkpoint(&self, messages: Vec<Message>) -> anyhow::Result<bool> {
        self.persist_recovery_checkpoint(&messages)
    }

    /// Durably enter one Phase 4 recovery round before planning or provider work.
    fn enter_phase_four_round(&self) -> anyhow::Result<bool> {
        Ok(true)
    }

    /// Durably record whether a committed Phase 4 retry freed five context points.
    fn record_phase_four_progress(&self, _meaningful: bool) -> anyhow::Result<()> {
        anyhow::bail!("Phase 4 progress persistence is unavailable")
    }

    /// Durably record a failed Phase 4 attempt, optionally disabling immediately.
    fn record_phase_four_failure(&self, _disable: bool) -> anyhow::Result<()> {
        anyhow::bail!("Phase 4 failure persistence is unavailable")
    }

    /// Charge one imminent Phase 4 provider attempt against this turn's input budget.
    fn charge_phase_four_side_call(&self, _estimated_input: u64) -> bool {
        true
    }

    /// Durably checkpoint one imminent Phase 4 provider attempt.
    fn checkpoint_phase_four_side_call(&self, estimated_input: u64) -> anyhow::Result<bool> {
        Ok(self.charge_phase_four_side_call(estimated_input))
    }

    /// Whether the active turn was cancelled while persistence was pending.
    fn phase_four_cancelled(&self) -> bool {
        false
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_injected_user_persistence_fails_closed() {
        let message = Message::injected_user("injection-1", "additional direction");

        assert!(NullSink.persist_injected_user(&message).is_err());
    }

    #[test]
    fn default_progress_checkpoint_fails_closed() {
        assert!(NullSink.record_phase_four_progress(true).is_err());
        assert!(NullSink.record_phase_four_failure(false).is_err());
    }
}
