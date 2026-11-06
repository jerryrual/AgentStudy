//! Durable, turn-scoped Phase 4 breaker state.

use std::sync::Mutex;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const BREAKER_STATE_VERSION: u64 = 1;
const MAX_ATTEMPTS: u64 = 100;
const STATE_INTEGER_MAX: u64 = i64::MAX as u64;

/// Versioned Phase 4 state checkpointed before provider work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PhaseFourBreakerState {
    version: u64,
    pub(crate) attempts: u64,
    pub(crate) consecutive_no_progress: u64,
    pub(crate) tail_override: bool,
    pub(crate) disabled: bool,
    pub(crate) side_call_tokens: u64,
}

impl Default for PhaseFourBreakerState {
    fn default() -> Self {
        Self {
            version: BREAKER_STATE_VERSION,
            attempts: 0,
            consecutive_no_progress: 0,
            tail_override: false,
            disabled: false,
            side_call_tokens: 0,
        }
    }
}

impl PhaseFourBreakerState {
    pub(crate) fn from_value(value: &Value) -> Option<Self> {
        let state = Self::deserialize(value).ok()?;
        (state.version == BREAKER_STATE_VERSION
            && state.attempts <= STATE_INTEGER_MAX
            && state.consecutive_no_progress <= STATE_INTEGER_MAX
            && state.side_call_tokens <= STATE_INTEGER_MAX)
            .then_some(state)
    }

    pub(crate) fn enter_round(&mut self, side_call_limit: i64) -> bool {
        let exhausted = self.disabled
            || self.attempts >= MAX_ATTEMPTS
            || (side_call_limit >= 0 && self.side_call_tokens >= side_call_limit as u64);
        if exhausted {
            self.disabled = true;
            return false;
        }
        self.attempts += 1;
        true
    }

    pub(crate) fn record_progress(&mut self, meaningful: bool) {
        if meaningful {
            self.consecutive_no_progress = 0;
            return;
        }
        self.consecutive_no_progress = self
            .consecutive_no_progress
            .saturating_add(1)
            .min(STATE_INTEGER_MAX);
        self.tail_override = true;
        self.disabled = self.disabled || self.consecutive_no_progress >= 2;
    }

    pub(crate) fn record_failure(&mut self, disable: bool) {
        self.record_progress(false);
        self.disabled = self.disabled || disable;
    }
}

/// Serializes breaker transitions for one logical user turn.
pub(crate) struct PhaseFourBreaker {
    limit: i64,
    state: Mutex<PhaseFourBreakerState>,
}

impl PhaseFourBreaker {
    pub(crate) fn new(limit: i64, state: PhaseFourBreakerState) -> Self {
        debug_assert!(limit >= -1);
        Self {
            limit,
            state: Mutex::new(state),
        }
    }

    pub(crate) fn snapshot(&self) -> Result<PhaseFourBreakerState> {
        self.state
            .lock()
            .map(|state| state.clone())
            .map_err(|_| anyhow::anyhow!("Phase 4 breaker lock is poisoned"))
    }

    pub(crate) fn enter_round_and_persist<F>(&self, persist: F) -> Result<bool>
    where
        F: FnOnce(&PhaseFourBreakerState, i64) -> Result<Option<(PhaseFourBreakerState, bool)>>,
    {
        let mut current = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Phase 4 breaker lock is poisoned"))?;
        let (committed, admitted) = persist(&current, self.limit)?
            .ok_or_else(|| anyhow::anyhow!("Phase 4 breaker checkpoint was declined"))?;
        *current = committed;
        Ok(admitted)
    }

    pub(crate) fn record_progress_and_persist<F>(&self, meaningful: bool, persist: F) -> Result<()>
    where
        F: FnOnce(&PhaseFourBreakerState, bool) -> Result<Option<PhaseFourBreakerState>>,
    {
        let mut current = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Phase 4 breaker lock is poisoned"))?;
        let committed = persist(&current, meaningful)?
            .ok_or_else(|| anyhow::anyhow!("Phase 4 breaker checkpoint was declined"))?;
        *current = committed;
        Ok(())
    }

    pub(crate) fn record_failure_and_persist<F>(&self, disable: bool, persist: F) -> Result<()>
    where
        F: FnOnce(&PhaseFourBreakerState, bool) -> Result<Option<PhaseFourBreakerState>>,
    {
        let mut current = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Phase 4 breaker lock is poisoned"))?;
        let committed = persist(&current, disable)?
            .ok_or_else(|| anyhow::anyhow!("Phase 4 breaker checkpoint was declined"))?;
        *current = committed;
        Ok(())
    }

    pub(crate) fn charge_and_persist<F>(&self, estimated_input: u64, persist: F) -> Result<bool>
    where
        F: FnOnce(
            &PhaseFourBreakerState,
            &PhaseFourBreakerState,
        ) -> Result<Option<PhaseFourBreakerState>>,
    {
        let mut current = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Phase 4 breaker lock is poisoned"))?;
        let mut charged = current.clone();
        charged.side_call_tokens = charged
            .side_call_tokens
            .saturating_add(estimated_input)
            .min(STATE_INTEGER_MAX);
        let committed = persist(&current, &charged)?
            .ok_or_else(|| anyhow::anyhow!("Phase 4 breaker checkpoint was declined"))?;
        let admitted = self.limit < 0 || committed.side_call_tokens < self.limit as u64;
        *current = committed;
        Ok(admitted)
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> PhaseFourBreakerState {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_entry_persists_attempts_and_disables_on_limits() {
        let mut state = PhaseFourBreakerState::default();
        for expected in 1..=MAX_ATTEMPTS {
            assert!(state.enter_round(-1));
            assert_eq!(state.attempts, expected);
        }
        assert!(!state.enter_round(-1));
        assert!(state.disabled);
        assert_eq!(state.attempts, MAX_ATTEMPTS);

        let mut spent = PhaseFourBreakerState {
            side_call_tokens: 100,
            ..PhaseFourBreakerState::default()
        };
        assert!(!spent.enter_round(100));
        assert!(spent.disabled);
        assert_eq!(spent.attempts, 0);
    }

    #[test]
    fn failure_strikes_support_immediate_budget_disable() {
        let mut generation = PhaseFourBreakerState::default();
        generation.record_failure(false);
        assert_eq!(generation.consecutive_no_progress, 1);
        assert!(generation.tail_override);
        assert!(!generation.disabled);

        let mut budget = PhaseFourBreakerState::default();
        budget.record_failure(true);
        assert_eq!(budget.consecutive_no_progress, 1);
        assert!(budget.tail_override);
        assert!(budget.disabled);
    }

    #[test]
    fn progress_resets_only_consecutive_strikes_and_second_strike_disables() {
        let mut state = PhaseFourBreakerState::default();
        state.record_progress(false);
        assert_eq!(state.consecutive_no_progress, 1);
        assert!(state.tail_override);
        assert!(!state.disabled);

        state.record_progress(true);
        assert_eq!(state.consecutive_no_progress, 0);
        assert!(state.tail_override);
        assert!(!state.disabled);

        state.record_progress(false);
        state.record_progress(false);
        assert_eq!(state.consecutive_no_progress, 2);
        assert!(state.tail_override);
        assert!(state.disabled);
    }

    #[test]
    fn charge_is_checkpointed_before_strict_admission() {
        let breaker = PhaseFourBreaker::new(100, PhaseFourBreakerState::default());
        assert!(breaker
            .charge_and_persist(40, |_, next| Ok(Some(next.clone())))
            .unwrap());
        assert!(!breaker
            .charge_and_persist(60, |_, next| Ok(Some(next.clone())))
            .unwrap());
        assert_eq!(breaker.state().side_call_tokens, 100);
    }

    #[test]
    fn declined_or_failed_checkpoint_does_not_mutate_memory() {
        let breaker = PhaseFourBreaker::new(-1, PhaseFourBreakerState::default());
        assert!(breaker.charge_and_persist(10, |_, _| Ok(None)).is_err());
        assert!(breaker
            .charge_and_persist(10, |_, _| anyhow::bail!("disk full"))
            .is_err());
        assert_eq!(breaker.state(), PhaseFourBreakerState::default());
    }

    #[test]
    fn state_validation_rejects_unsupported_or_oversized_data() {
        let valid = serde_json::to_value(PhaseFourBreakerState::default()).unwrap();
        assert_eq!(
            PhaseFourBreakerState::from_value(&valid),
            Some(PhaseFourBreakerState::default())
        );
        let mut unsupported = valid.clone();
        unsupported["version"] = Value::from(2);
        assert!(PhaseFourBreakerState::from_value(&unsupported).is_none());
        let mut oversized = valid;
        oversized["side_call_tokens"] = Value::from(u64::MAX);
        assert!(PhaseFourBreakerState::from_value(&oversized).is_none());
    }
}
