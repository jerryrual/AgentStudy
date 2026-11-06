//! Atomic delivery of one same-turn injected user message.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::model::Message;

/// Create the producer and consumer handles for one agent turn.
pub fn channel() -> (InjectionQueue, InjectionSource) {
    let inner = Arc::new(Inner {
        state: Mutex::new(State {
            accepting: true,
            slot: Slot::Empty,
        }),
    });
    (
        InjectionQueue {
            inner: inner.clone(),
        },
        InjectionSource { inner },
    )
}

#[derive(Debug)]
struct Inner {
    state: Mutex<State>,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[derive(Debug)]
struct State {
    accepting: bool,
    slot: Slot,
}

#[derive(Debug)]
enum Slot {
    Empty,
    Pending(Message),
    Reserved,
}

/// The producer half held by the turn's input owner.
#[derive(Clone, Debug)]
pub struct InjectionQueue {
    inner: Arc<Inner>,
}

impl InjectionQueue {
    /// Queue one same-turn user message.
    ///
    /// The rejected message is returned intact so the caller can retain it for
    /// a later turn instead of silently losing user input.
    pub fn enqueue(
        &self,
        injection_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<(), EnqueueError> {
        let message = Message::injected_user(injection_id, content);
        let mut state = self.inner.lock();
        if !state.accepting {
            return Err(EnqueueError::Closed(message));
        }
        if !matches!(state.slot, Slot::Empty) {
            return Err(EnqueueError::Full(message));
        }
        state.slot = Slot::Pending(message);
        Ok(())
    }

    /// Cancel the matching message while it is still pending.
    ///
    /// A reserved message has crossed the consumer boundary and cannot be
    /// cancelled. The returned message can be restored to the input editor.
    pub fn cancel(&self, injection_id: &str) -> Option<Message> {
        let mut state = self.inner.lock();
        let matches = matches!(
            &state.slot,
            Slot::Pending(Message::InjectedUser {
                injection_id: pending_id,
                ..
            }) if pending_id == injection_id
        );
        if !matches {
            return None;
        }
        match std::mem::replace(&mut state.slot, Slot::Empty) {
            Slot::Pending(message) => Some(message),
            _ => unreachable!("the matching pending slot changed while locked"),
        }
    }

    /// Stop accepting new messages without discarding a pending message.
    pub fn close(&self) {
        self.inner.lock().accepting = false;
    }
}

/// Why an injected message was not accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnqueueError {
    /// The one available slot is pending or reserved.
    Full(Message),
    /// The active turn no longer accepts injections.
    Closed(Message),
}

impl EnqueueError {
    /// Recover the exact message that the queue rejected.
    pub fn into_message(self) -> Message {
        match self {
            Self::Full(message) | Self::Closed(message) => message,
        }
    }
}

impl fmt::Display for EnqueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full(_) => formatter.write_str("the injection slot is full"),
            Self::Closed(_) => formatter.write_str("the injection queue is closed"),
        }
    }
}

impl std::error::Error for EnqueueError {}

/// The consumer half owned by the running agent.
#[derive(Debug)]
pub struct InjectionSource {
    inner: Arc<Inner>,
}

impl InjectionSource {
    /// Reserve the pending message for the next provider request.
    ///
    /// Dropping the reservation rolls it back into the slot. Call `commit`
    /// after the transcript sink accepts the message; the agent crosses its
    /// strict recovery barrier immediately afterward.
    pub(crate) fn reserve(&self) -> Option<InjectionReservation> {
        let mut state = self.inner.lock();
        let message = match std::mem::replace(&mut state.slot, Slot::Reserved) {
            Slot::Pending(message) => message,
            slot => {
                state.slot = slot;
                return None;
            }
        };
        Some(InjectionReservation {
            inner: self.inner.clone(),
            message: Some(message),
        })
    }

    /// Seal and finish the source if no accepted message remains.
    ///
    /// This shares the enqueue lock, so an enqueue racing a tool-free final
    /// response either wins and keeps the turn alive, or observes the seal and
    /// returns its message to the producer.
    pub(crate) fn try_finish(&self) -> bool {
        let mut state = self.inner.lock();
        if !matches!(state.slot, Slot::Empty) {
            return false;
        }
        state.accepting = false;
        true
    }

    pub(crate) fn seal_on_drop(&self) -> SealOnDrop<'_> {
        SealOnDrop(self)
    }

    fn seal(&self) {
        self.inner.lock().accepting = false;
    }
}

pub(crate) struct SealOnDrop<'a>(&'a InjectionSource);

impl Drop for SealOnDrop<'_> {
    fn drop(&mut self) {
        self.0.seal();
    }
}

/// A pending injection borrowed for one provider boundary.
#[derive(Debug)]
pub(crate) struct InjectionReservation {
    inner: Arc<Inner>,
    message: Option<Message>,
}

impl InjectionReservation {
    pub(crate) fn message(&self) -> &Message {
        self.message
            .as_ref()
            .expect("an active reservation always owns its message")
    }

    /// Consume the slot and return its message to the agent.
    pub(crate) fn commit(mut self) -> Message {
        let mut state = self.inner.lock();
        assert!(
            matches!(state.slot, Slot::Reserved),
            "an active reservation must own the reserved slot"
        );
        state.slot = Slot::Empty;
        self.message
            .take()
            .expect("an active reservation always owns its message")
    }
}

impl Drop for InjectionReservation {
    fn drop(&mut self) {
        let Some(message) = self.message.take() else {
            return;
        };
        let mut state = self.inner.lock();
        assert!(
            matches!(state.slot, Slot::Reserved),
            "a dropped reservation must own the reserved slot"
        );
        state.slot = Slot::Pending(message);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

    fn assert_injection(message: &Message, expected_id: &str, expected_content: &str) {
        assert!(matches!(
            message,
            Message::InjectedUser {
                injection_id,
                content,
            } if injection_id == expected_id && content == expected_content
        ));
    }

    #[test]
    fn enqueue_and_commit_deliver_an_injected_user_message() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "new direction").unwrap();

        let reservation = source.reserve().expect("pending injection");
        assert_injection(reservation.message(), "inject-1", "new direction");
        let message = reservation.commit();

        assert_injection(&message, "inject-1", "new direction");
        assert!(source.reserve().is_none());
    }

    #[test]
    fn the_single_slot_rejects_another_message_intact() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "first").unwrap();

        let error = queue.enqueue("inject-2", "second").unwrap_err();
        assert!(matches!(error, EnqueueError::Full(_)));
        assert_injection(&error.into_message(), "inject-2", "second");

        assert_injection(source.reserve().unwrap().message(), "inject-1", "first");
    }

    #[test]
    fn a_reserved_slot_remains_full_until_commit() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "first").unwrap();
        let reservation = source.reserve().unwrap();

        assert!(matches!(
            queue.enqueue("inject-2", "second"),
            Err(EnqueueError::Full(_))
        ));
        reservation.commit();
        queue.enqueue("inject-3", "third").unwrap();
        assert_injection(source.reserve().unwrap().message(), "inject-3", "third");
    }

    #[test]
    fn dropping_a_reservation_rolls_the_message_back() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "retry me").unwrap();

        let first = source.reserve().unwrap();
        assert_injection(first.message(), "inject-1", "retry me");
        drop(first);

        let second = source.reserve().expect("rolled-back injection");
        assert_injection(second.message(), "inject-1", "retry me");
        second.commit();
        assert!(source.reserve().is_none());
    }

    #[test]
    fn cancel_only_removes_the_matching_pending_message() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "cancel me").unwrap();

        assert!(queue.cancel("inject-other").is_none());
        let cancelled = queue.cancel("inject-1").expect("matching injection");
        assert_injection(&cancelled, "inject-1", "cancel me");
        assert!(source.reserve().is_none());

        queue.enqueue("inject-2", "replacement").unwrap();
        assert_injection(
            source.reserve().unwrap().message(),
            "inject-2",
            "replacement",
        );
    }

    #[test]
    fn a_reserved_message_cannot_be_cancelled() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "in flight").unwrap();
        let reservation = source.reserve().unwrap();

        assert!(queue.cancel("inject-1").is_none());
        assert_injection(&reservation.commit(), "inject-1", "in flight");
    }

    #[test]
    fn close_rejects_new_messages_but_preserves_the_pending_slot() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "already accepted").unwrap();
        queue.close();

        let error = queue.enqueue("inject-2", "too late").unwrap_err();
        assert!(matches!(error, EnqueueError::Closed(_)));
        assert_injection(&error.into_message(), "inject-2", "too late");
        assert_injection(
            &source.reserve().unwrap().commit(),
            "inject-1",
            "already accepted",
        );
        assert!(source.try_finish());
    }

    #[test]
    fn dropping_the_source_guard_seals_admission_and_preserves_pending_input() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "already accepted").unwrap();

        drop(source.seal_on_drop());

        assert!(matches!(
            queue.enqueue("inject-2", "too late"),
            Err(EnqueueError::Closed(_))
        ));
        assert_injection(
            source.reserve().unwrap().message(),
            "inject-1",
            "already accepted",
        );
    }

    #[test]
    fn try_finish_refuses_pending_and_reserved_messages() {
        let (queue, source) = channel();
        queue.enqueue("inject-1", "pending").unwrap();
        assert!(!source.try_finish());

        let reservation = source.reserve().unwrap();
        assert!(!source.try_finish());
        drop(reservation);
        assert!(!source.try_finish());

        source.reserve().unwrap().commit();
        assert!(source.try_finish());
    }

    #[test]
    fn finishing_an_empty_source_atomically_seals_enqueue() {
        let (queue, source) = channel();
        assert!(source.try_finish());

        let error = queue.enqueue("inject-1", "too late").unwrap_err();
        assert!(matches!(error, EnqueueError::Closed(_)));
        assert_injection(&error.into_message(), "inject-1", "too late");
        assert!(source.try_finish());
    }

    #[test]
    fn enqueue_and_finish_race_without_accepting_lost_input() {
        for iteration in 0..256 {
            let (queue, source) = channel();
            let barrier = Arc::new(Barrier::new(2));
            let producer_barrier = barrier.clone();
            let producer = std::thread::spawn(move || {
                producer_barrier.wait();
                queue.enqueue(format!("inject-{iteration}"), "racing input")
            });

            barrier.wait();
            let finished = source.try_finish();
            let enqueue = producer.join().unwrap();

            match (finished, enqueue) {
                (true, Err(EnqueueError::Closed(message))) => {
                    assert_injection(&message, &format!("inject-{iteration}"), "racing input");
                    assert!(source.reserve().is_none());
                }
                (false, Ok(())) => {
                    let reservation = source.reserve().expect("accepted racing input");
                    assert_injection(
                        reservation.message(),
                        &format!("inject-{iteration}"),
                        "racing input",
                    );
                    reservation.commit();
                    assert!(source.try_finish());
                }
                outcome => panic!("enqueue/finish did not linearize: {outcome:?}"),
            }
        }
    }
}
