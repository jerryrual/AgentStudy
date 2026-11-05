//! Model context-window budgeting and deterministic token estimates.

use serde_json::Value;

/// Compaction thresholds derived from one model profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub safety_margin: u32,
    pub trigger_tokens: u32,
    pub target_tokens: u32,
    pub force_tokens: u32,
}

impl ContextBudget {
    /// Derive iCode-compatible trigger, target and force thresholds.
    pub fn derive(context_window: u32, max_output_tokens: u32) -> Option<Self> {
        if context_window < 100 || max_output_tokens == 0 || max_output_tokens >= context_window {
            return None;
        }
        let safety_margin = 5_000.max(ceil_percent(context_window, 5));
        let trigger_tokens = context_window
            .saturating_sub(max_output_tokens.saturating_add(safety_margin))
            .max(ceil_percent(context_window, 30));
        let mut target_tokens =
            round_ratio(trigger_tokens, 5, 8).clamp(1, trigger_tokens.saturating_sub(2));
        let force_tokens = round_ratio(trigger_tokens, 7, 8).clamp(
            target_tokens.saturating_add(1),
            trigger_tokens.saturating_sub(1),
        );
        target_tokens = target_tokens.min(force_tokens.saturating_sub(1));
        Some(Self {
            context_window,
            max_output_tokens,
            safety_margin,
            trigger_tokens,
            target_tokens,
            force_tokens,
        })
    }
}

/// Derive context thresholds without constructing [`ContextBudget`] directly.
pub fn derive_context_budget(context_window: u32, max_output_tokens: u32) -> Option<ContextBudget> {
    ContextBudget::derive(context_window, max_output_tokens)
}

/// Conservatively estimate at most one token per UTF-8 byte.
pub fn estimate_tokens(text: &str) -> u32 {
    text.len().min(u32::MAX as usize) as u32
}

/// Estimate the serialized size of a JSON request in tokens.
pub fn estimate_json_tokens(value: &Value) -> u32 {
    estimate_tokens(&value.to_string())
}

/// Clamp an output cap to the estimated room left in a context window.
///
/// A one-percent (minimum 256-token) margin is retained when possible. The
/// margin yields before reducing useful output below 1,024 tokens.
pub fn clamp_output_tokens(
    context_window: u32,
    configured_output: u32,
    estimated_input: u32,
) -> u32 {
    let room = context_window.saturating_sub(estimated_input);
    if room == 0 || configured_output == 0 {
        return 0;
    }
    let desired_margin = 256.max(ceil_percent(context_window, 1));
    let useful_output = configured_output.min(1_024).min(room);
    let margin = desired_margin.min(room.saturating_sub(useful_output));
    configured_output.min(room.saturating_sub(margin))
}

fn round_ratio(value: u32, numerator: u32, denominator: u32) -> u32 {
    ((u64::from(value) * u64::from(numerator) + u64::from(denominator / 2))
        / u64::from(denominator)) as u32
}

fn ceil_percent(value: u32, percent: u32) -> u32 {
    div_ceil(u64::from(value) * u64::from(percent), 100) as u32
}

fn div_ceil(value: u64, divisor: u64) -> u64 {
    value.div_ceil(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn derives_deepseek_thresholds() {
        let budget = ContextBudget::derive(128_000, 8_192).unwrap();
        assert_eq!(budget.safety_margin, 6_400);
        assert_eq!(budget.trigger_tokens, 113_408);
        assert_eq!(budget.target_tokens, 70_880);
        assert_eq!(budget.force_tokens, 99_232);
        assert!(budget.target_tokens < budget.force_tokens);
        assert!(budget.force_tokens < budget.trigger_tokens);
    }

    #[test]
    fn trigger_has_a_thirty_percent_floor_and_half_up_rounding() {
        let budget = derive_context_budget(10_000, 6_000).unwrap();
        assert_eq!(budget.safety_margin, 5_000);
        assert_eq!(budget.trigger_tokens, 3_000);
        assert_eq!(budget.target_tokens, 1_875);
        assert_eq!(budget.force_tokens, 2_625);

        let fractional = derive_context_budget(103, 1).unwrap();
        assert_eq!(fractional.trigger_tokens, 31);
        assert_eq!(fractional.target_tokens, 19);
        assert_eq!(fractional.force_tokens, 27);
        assert!(ContextBudget::derive(99, 1).is_none());
        assert!(ContextBudget::derive(8_192, 8_192).is_none());
    }

    #[test]
    fn estimates_ascii_cjk_and_json_conservatively() {
        assert_eq!(estimate_tokens("abcdefgh"), 8);
        let adversarial = r#"{}[]!@#$%^&*()_+-=ABC_123_xyz"#;
        assert_eq!(estimate_tokens(adversarial), adversarial.len() as u32);
        assert_eq!(estimate_tokens("项目记忆"), 12);
        let value = json!({"text": "abcdefgh"});
        assert_eq!(estimate_json_tokens(&value), value.to_string().len() as u32);
    }

    #[test]
    fn clamps_output_to_context_room_and_yields_margin() {
        assert_eq!(clamp_output_tokens(10_000, 2_000, 7_000), 2_000);
        assert_eq!(clamp_output_tokens(10_000, 2_000, 8_500), 1_244);
        assert_eq!(clamp_output_tokens(10_000, 2_000, 9_500), 500);
        assert_eq!(clamp_output_tokens(10_000, 2_000, 10_000), 0);
        assert_eq!(clamp_output_tokens(10_000, 0, 1_000), 0);
    }
}
