//! Durable, turn-scoped Phase 4 breaker state.

use std::sync::Mutex;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const BREAKER_STATE_VERSION: u64 = 1;
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
        let state: Self = serde_json::from_value(value.clone()).ok()?;
        (state.version == BREAKER_STATE_VERSION
            && state.attempts <= STATE_INTEGER_MAX
            && state.consecutive_no_progress <= STATE_INTEGER_MAX
            && state.side_call_tokens <= STATE_INTEGER_MAX)
            .then_some(state)
    }
}

/// Serializes charge-first state checkpoints for one logical user turn.
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

    pub(crate) fn charge_and_persist<F>(&self, estimated_input: u64, persist: F) -> Result<bool>
    where
        F: FnOnce(&PhaseFourBreakerState) -> Result<bool>,
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
        if !persist(&charged)? {
            bail!("Phase 4 breaker checkpoint was declined");
        }
        let admitted = self.limit < 0 || charged.side_call_tokens < self.limit as u64;
        *current = charged;
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
    fn charge_is_checkpointed_before_strict_admission() {
        let breaker = PhaseFourBreaker::new(100, PhaseFourBreakerState::default());
        assert!(breaker.charge_and_persist(40, |_| Ok(true)).unwrap());
        assert!(!breaker.charge_and_persist(60, |_| Ok(true)).unwrap());
        assert_eq!(breaker.state().side_call_tokens, 100);
    }

    #[test]
    fn declined_or_failed_checkpoint_does_not_mutate_memory() {
        let breaker = PhaseFourBreaker::new(-1, PhaseFourBreakerState::default());
        assert!(breaker.charge_and_persist(10, |_| Ok(false)).is_err());
        assert!(breaker
            .charge_and_persist(10, |_| anyhow::bail!("disk full"))
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
