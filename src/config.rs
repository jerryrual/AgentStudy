//! Model profiles, user settings and API-key resolution.
//!
//! Configuration lives next to the iCode TUI configuration, but in a single
//! JSON document (`$ICODE_HOME/settings.json`, or the platform config
//! directory). A profile is one OpenAI-compatible endpoint plus the model and
//! token budget to use with it.
//!
//! API keys may be written inline, referenced from an environment variable
//! (`{{DEEPSEEK_API_KEY}}`) or fetched from a command (`{{!token-command}}`),
//! mirroring iCode's credential handling.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Deserializer, Serialize};

/// Name of the built-in profile created on first run.
pub const DEFAULT_PROFILE: &str = "deepseek";

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_LAST_WORDS_SUPPLEMENT_CHARS: usize = 4_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Ask the user before every mutating operation.
    #[default]
    Ask,
    /// Approve every mutating operation automatically.
    Auto,
    /// Refuse every mutating operation.
    Deny,
}

impl ApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::Auto => "auto",
            Self::Deny => "deny",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ask" | "always" => Some(Self::Ask),
            "auto" | "allow" | "never" => Some(Self::Auto),
            "deny" | "readonly" | "read-only" => Some(Self::Deny),
            _ => None,
        }
    }
}

impl std::fmt::Display for ApprovalMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ApiStyle {
    /// `POST {base}/chat/completions` with SSE streaming.
    #[default]
    Chat,
    /// OpenAI-compatible Responses API.
    Responses,
}

impl ApiStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Display name used by `--model`, `/model` and the status bar.
    pub name: String,
    /// Provider identifier; `deepseek` and `openai` have known defaults.
    #[serde(default)]
    pub provider: String,
    /// Which wire API to speak to the provider.
    #[serde(default)]
    pub api_style: ApiStyle,
    /// Provider model identifier, e.g. `deepseek-chat`.
    pub model: String,
    /// Base URL; empty means the provider default.
    #[serde(default)]
    pub base_url: String,
    /// API key, an `{{ENV_VAR}}` reference or a `{{!command}}` command.
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Total model context capacity in tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional trusted-local emphasis appended to Phase 4 Last Words instructions.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_words_supplement: String,
    /// Stream responses token-by-token. Enabled by default.
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_stream() -> bool {
    true
}

impl ModelProfile {
    /// The built-in DeepSeek profile used when no configuration exists yet.
    pub fn builtin_deepseek() -> Self {
        Self {
            name: DEFAULT_PROFILE.to_string(),
            provider: "deepseek".to_string(),
            api_style: ApiStyle::Chat,
            model: "deepseek-chat".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            api_key: "{{DEEPSEEK_API_KEY}}".to_string(),
            max_tokens: Some(8192),
            context_window: Some(128_000),
            temperature: Some(0.1),
            last_words_supplement: String::new(),
            stream: true,
        }
    }

    /// Resolve the base URL that requests are sent to.
    pub fn resolved_base_url(&self) -> String {
        let base = if self.base_url.trim().is_empty() {
            default_base_url(&self.provider).to_string()
        } else {
            self.base_url.trim().to_string()
        };
        base.trim_end_matches('/').to_string()
    }

    /// Full chat-completions endpoint.
    pub fn chat_completions_url(&self) -> String {
        let base = self.resolved_base_url();
        if base.ends_with("/chat/completions") {
            base
        } else {
            format!("{base}/chat/completions")
        }
    }

    /// Full Responses API endpoint.
    pub fn responses_url(&self) -> String {
        let base = self.resolved_base_url();
        if base.ends_with("/responses") {
            base
        } else {
            format!("{base}/responses")
        }
    }

    /// Human-readable description of where the API key comes from.
    pub fn key_source(&self) -> String {
        let raw = self.api_key.trim();
        if let Some(inner) = placeholder_inner(raw) {
            if let Some(command) = inner.strip_prefix('!') {
                return format!("command: {}", command.trim());
            }
            return format!("env: {inner}");
        }
        if raw.is_empty() {
            "not configured".to_string()
        } else {
            "inline".to_string()
        }
    }
}

/// Default endpoint for a provider identifier.
pub fn default_base_url(provider: &str) -> &'static str {
    match provider.trim().to_ascii_lowercase().as_str() {
        "deepseek" => "https://api.deepseek.com",
        "openai" => "https://api.openai.com/v1",
        _ => "https://api.openai.com/v1",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Configured model endpoints.
    #[serde(default)]
    pub models: Vec<ModelProfile>,
    /// Profile used when `--model` is not given.
    #[serde(default)]
    pub current_model: Option<String>,
    /// Default approval policy for mutating tools.
    #[serde(default)]
    pub approval_mode: ApprovalMode,
    /// Default agent (`code` or `qa`).
    #[serde(default = "default_agent")]
    pub agent: String,
    /// Maximum aggregate estimated Phase 4 side-call input tokens per turn.
    /// `-1` is unlimited and `0` disables Phase 4 side calls.
    #[serde(
        default = "default_phase4_side_call_token_budget",
        deserialize_with = "deserialize_phase4_side_call_token_budget"
    )]
    pub phase4_side_call_token_budget: i64,
    /// Maximum number of model calls in one user turn.
    #[serde(default = "default_max_rounds")]
    pub max_tool_rounds: usize,
}

fn default_agent() -> String {
    "code".to_string()
}

fn default_phase4_side_call_token_budget() -> i64 {
    -1
}

fn deserialize_phase4_side_call_token_budget<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<i64>::deserialize(deserializer)?
        .unwrap_or_else(default_phase4_side_call_token_budget))
}

fn default_max_rounds() -> usize {
    40
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            models: vec![ModelProfile::builtin_deepseek()],
            current_model: Some(DEFAULT_PROFILE.to_string()),
            approval_mode: ApprovalMode::Ask,
            agent: default_agent(),
            phase4_side_call_token_budget: default_phase4_side_call_token_budget(),
            max_tool_rounds: default_max_rounds(),
        }
    }
}

impl Settings {
    /// Directory that holds `settings.json` and sessions.
    pub fn config_dir() -> PathBuf {
        if let Some(home) = std::env::var_os("ICODE_HOME") {
            return PathBuf::from(home);
        }
        dirs::config_dir()
            .map(|dir| dir.join("icode"))
            .unwrap_or_else(|| PathBuf::from(".icode"))
    }

    /// Directory that holds session transcripts.
    pub fn data_dir() -> PathBuf {
        if let Some(home) = std::env::var_os("ICODE_HOME") {
            return PathBuf::from(home);
        }
        dirs::data_local_dir()
            .map(|dir| dir.join("icode"))
            .unwrap_or_else(|| PathBuf::from(".icode"))
    }

    pub fn path() -> PathBuf {
        Self::config_dir().join("settings.json")
    }

    /// Load settings, creating a default document on first run.
    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.exists() {
            let settings = Self::default();
            settings.save_to(&path)?;
            return Ok(settings);
        }
        let mut settings = Self::load_from(&path)?;
        settings.normalise()?;
        Ok(settings)
    }

    /// Load settings from an explicit path (used by tests).
    pub fn load_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let mut settings: Settings = serde_json::from_str(&text)
            .with_context(|| format!("invalid settings file {}", path.display()))?;
        settings.normalise()?;
        Ok(settings)
    }

    /// Validate the document and fill in defaults for missing pieces.
    pub fn normalise(&mut self) -> Result<()> {
        if self.phase4_side_call_token_budget < -1 {
            bail!("phase4_side_call_token_budget must be -1 or greater");
        }
        if self.models.is_empty() {
            self.models.push(ModelProfile::builtin_deepseek());
        }
        for profile in &self.models {
            if profile.name.trim().is_empty() {
                bail!("a model profile has an empty name");
            }
            if profile.model.trim().is_empty() {
                bail!("model profile {:?} has an empty model id", profile.name);
            }
            if profile.last_words_supplement.chars().count() > MAX_LAST_WORDS_SUPPLEMENT_CHARS {
                bail!(
                    "model profile {:?} Last Words supplement exceeds {MAX_LAST_WORDS_SUPPLEMENT_CHARS} characters",
                    profile.name
                );
            }
            if let Some(context_window) = profile.context_window {
                if context_window == 0 {
                    bail!("model profile {:?} has a zero context window", profile.name);
                }
                if let Some(max_tokens) = profile.max_tokens {
                    if context_window <= max_tokens {
                        bail!(
                            "model profile {:?} context window ({context_window}) must exceed \
                             max tokens ({max_tokens})",
                            profile.name
                        );
                    }
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for profile in &self.models {
            if !seen.insert(profile.name.to_ascii_lowercase()) {
                bail!("duplicate model profile name {:?}", profile.name);
            }
        }
        let known = self.models.iter().any(|profile| {
            self.current_model
                .as_deref()
                .is_some_and(|current| profile.name == current)
        });
        if !known {
            self.current_model = Some(self.models[0].name.clone());
        }
        if self.max_tool_rounds == 0 {
            self.max_tool_rounds = default_max_rounds();
        }
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::path())
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)
            .with_context(|| format!("failed to write settings file {}", path.display()))?;
        Ok(())
    }

    /// Look up a profile by name, or return the current/default profile.
    pub fn profile(&self, name: Option<&str>) -> Result<ModelProfile> {
        if let Some(name) = name {
            return self
                .models
                .iter()
                .find(|profile| profile.name == name)
                .cloned()
                .with_context(|| {
                    let available: Vec<&str> =
                        self.models.iter().map(|p| p.name.as_str()).collect();
                    format!(
                        "unknown model {name:?}; configured models: {}",
                        available.join(", ")
                    )
                });
        }
        if let Some(current) = self.current_model.as_deref() {
            if let Some(profile) = self.models.iter().find(|profile| profile.name == current) {
                return Ok(profile.clone());
            }
        }
        self.models
            .first()
            .cloned()
            .context("no model profiles are configured; edit settings.json")
    }
}

/// Resolve an API-key field into a concrete secret.
///
/// * `{{NAME}}` reads the environment variable `NAME`.
/// * `{{!cmd}}` runs `cmd` through the shell and uses trimmed stdout.
/// * anything else is returned verbatim.
pub fn resolve_api_key(raw: &str) -> Result<String> {
    let raw = raw.trim();
    if let Some(inner) = placeholder_inner(raw) {
        if let Some(command) = inner.strip_prefix('!') {
            return run_secret_command(command.trim());
        }
        let value = std::env::var(inner).with_context(|| {
            format!("environment variable {inner} is not set (referenced by the API key)")
        })?;
        if value.trim().is_empty() {
            bail!("environment variable {inner} is empty");
        }
        return Ok(value.trim().to_string());
    }
    if raw.is_empty() {
        bail!(
            "no API key configured; set one in settings.json or export the provider key \
             (for the built-in profile: DEEPSEEK_API_KEY)"
        );
    }
    Ok(raw.to_string())
}

fn placeholder_inner(raw: &str) -> Option<&str> {
    raw.strip_prefix("{{")
        .and_then(|rest| rest.strip_suffix("}}"))
        .map(str::trim)
}

fn run_secret_command(command: &str) -> Result<String> {
    use std::process::{Command, Stdio};
    use std::sync::mpsc;

    if command.is_empty() {
        bail!("API key command is empty");
    }
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to run API key command {command:?}"))?;

    // Drain both pipes on separate threads so a chatty command cannot block on
    // a full pipe buffer while we poll for completion.
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    let (sender, receiver) = mpsc::channel();
    let stdout_sender = sender.clone();
    std::thread::spawn(move || {
        let _ = stdout_sender.send((Stream::Stdout, read_capped(stdout)));
    });
    let stderr_sender = sender;
    std::thread::spawn(move || {
        let _ = stderr_sender.send((Stream::Stderr, read_capped(stderr)));
    });

    let started = Instant::now();
    let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = collect_output(&receiver);
                    bail!("API key command {command:?} timed out after {DEFAULT_TIMEOUT_SECS}s");
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = collect_output(&receiver);
                bail!("failed to wait for API key command {command:?}: {error}");
            }
        }
    };

    let output = collect_output(&receiver);
    if output.truncated {
        bail!(
            "API key command {command:?} produced more than {} KiB of output",
            SECRET_OUTPUT_CAP / 1024
        );
    }
    if !status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "API key command {command:?} exited with {status}: {}",
            stderr.trim()
        );
    }
    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if key.is_empty() {
        bail!("API key command {command:?} produced no output");
    }
    Ok(key)
}

/// Bytes captured per stream from an API-key command.
const SECRET_OUTPUT_CAP: usize = 64 * 1024;
/// Upper bound for waiting for inherited pipe handles to close.
const READER_JOIN_TIMEOUT: Duration = Duration::from_millis(750);

enum Stream {
    Stdout,
    Stderr,
}

#[derive(Default)]
struct CapturedOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    truncated: bool,
}

/// Collect both reader results, but never wait past [`READER_JOIN_TIMEOUT`]:
/// a background process can keep an inherited pipe open after the shell exits.
fn collect_output(
    receiver: &std::sync::mpsc::Receiver<(Stream, (Vec<u8>, bool))>,
) -> CapturedOutput {
    let mut output = CapturedOutput::default();
    let deadline = Instant::now() + READER_JOIN_TIMEOUT;
    for _ in 0..2 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Ok((stream, (bytes, truncated))) = receiver.recv_timeout(remaining) else {
            break;
        };
        match stream {
            Stream::Stdout => output.stdout = bytes,
            Stream::Stderr => output.stderr = bytes,
        }
        output.truncated |= truncated;
    }
    output
}

fn read_capped(mut reader: impl std::io::Read) -> (Vec<u8>, bool) {
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                if kept.len() < SECRET_OUTPUT_CAP {
                    let take = (SECRET_OUTPUT_CAP - kept.len()).min(read);
                    kept.extend_from_slice(&buffer[..take]);
                    if take < read {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
        }
    }
    (kept, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_have_one_usable_profile() {
        let settings = Settings::default();
        assert_eq!(settings.models.len(), 1);
        assert_eq!(settings.current_model.as_deref(), Some(DEFAULT_PROFILE));
        assert_eq!(settings.approval_mode, ApprovalMode::Ask);
        assert_eq!(settings.agent, "code");
        assert_eq!(settings.models[0].context_window, Some(128_000));
    }

    #[test]
    fn phase_four_side_call_budget_defaults_and_validates() {
        assert_eq!(Settings::default().phase4_side_call_token_budget, -1);

        let legacy: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy.phase4_side_call_token_budget, -1);
        let null_budget: Settings =
            serde_json::from_str(r#"{"phase4_side_call_token_budget":null}"#).unwrap();
        assert_eq!(null_budget.phase4_side_call_token_budget, -1);
        assert!(
            serde_json::from_str::<Settings>(r#"{"phase4_side_call_token_budget":true}"#).is_err()
        );

        let mut disabled = Settings {
            phase4_side_call_token_budget: 0,
            ..Default::default()
        };
        disabled.normalise().unwrap();
        assert_eq!(
            serde_json::to_value(disabled).unwrap()["phase4_side_call_token_budget"],
            0
        );

        let mut invalid = Settings {
            phase4_side_call_token_budget: -2,
            ..Default::default()
        };
        assert!(invalid.normalise().is_err());
    }

    #[test]
    fn context_window_is_backward_compatible_and_skipped_when_unknown() {
        let profile: ModelProfile = serde_json::from_str(
            r#"{
                "name": "legacy",
                "model": "legacy-model",
                "max_tokens": 1024
            }"#,
        )
        .unwrap();
        assert_eq!(profile.context_window, None);

        let json = serde_json::to_value(profile).unwrap();
        assert_eq!(json.get("context_window"), None);
    }

    #[test]
    fn last_words_supplement_is_backward_compatible_and_bounded() {
        let profile: ModelProfile = serde_json::from_str(
            r#"{
                "name": "legacy",
                "model": "legacy-model"
            }"#,
        )
        .unwrap();
        assert!(profile.last_words_supplement.is_empty());
        assert_eq!(
            serde_json::to_value(profile)
                .unwrap()
                .get("last_words_supplement"),
            None
        );

        let mut settings = Settings::default();
        settings.models[0].last_words_supplement = "x".repeat(MAX_LAST_WORDS_SUPPLEMENT_CHARS + 1);
        let error = settings.normalise().unwrap_err().to_string();
        assert!(error.contains("Last Words supplement exceeds"));

        settings.models[0].last_words_supplement = "排".repeat(MAX_LAST_WORDS_SUPPLEMENT_CHARS);
        settings.normalise().unwrap();
        assert_eq!(
            serde_json::to_value(&settings).unwrap()["models"][0]["last_words_supplement"],
            settings.models[0].last_words_supplement
        );
    }

    #[test]
    fn normalise_rejects_invalid_context_windows() {
        let mut settings = Settings::default();
        settings.models[0].context_window = Some(0);
        let error = settings.normalise().unwrap_err().to_string();
        assert!(error.contains("zero context window"));

        settings.models[0].context_window = settings.models[0].max_tokens;
        let error = settings.normalise().unwrap_err().to_string();
        assert!(error.contains("must exceed max tokens"));

        settings.models[0].context_window = Some(8193);
        settings.normalise().unwrap();
    }

    #[test]
    fn chat_url_is_derived_from_base() {
        let mut profile = ModelProfile::builtin_deepseek();
        assert_eq!(
            profile.chat_completions_url(),
            "https://api.deepseek.com/chat/completions"
        );
        profile.base_url = "http://localhost:8080/v1/".to_string();
        assert_eq!(
            profile.chat_completions_url(),
            "http://localhost:8080/v1/chat/completions"
        );
        profile.base_url = "http://localhost:8080/v1/chat/completions".to_string();
        assert_eq!(
            profile.chat_completions_url(),
            "http://localhost:8080/v1/chat/completions"
        );
    }

    #[test]
    fn responses_url_is_derived_from_base() {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.base_url = "https://api.openai.com/v1/".to_string();
        assert_eq!(
            profile.responses_url(),
            "https://api.openai.com/v1/responses"
        );
        profile.base_url = "http://localhost:8080/v1/responses".to_string();
        assert_eq!(
            profile.responses_url(),
            "http://localhost:8080/v1/responses"
        );
    }

    #[test]
    fn key_source_describes_placeholders() {
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "{{DEEPSEEK_API_KEY}}".to_string();
        assert_eq!(profile.key_source(), "env: DEEPSEEK_API_KEY");
        profile.api_key = "{{!token get}}".to_string();
        assert_eq!(profile.key_source(), "command: token get");
        profile.api_key = "sk-inline".to_string();
        assert_eq!(profile.key_source(), "inline");
    }

    #[test]
    fn env_placeholder_resolution() {
        // SAFETY: tests run in the same process; use a unique variable name.
        std::env::set_var("ICODE_TEST_KEY_1", "secret-1");
        assert_eq!(resolve_api_key("{{ICODE_TEST_KEY_1}}").unwrap(), "secret-1");
        assert!(resolve_api_key("{{ICODE_TEST_MISSING_KEY}}").is_err());
        assert_eq!(resolve_api_key("inline").unwrap(), "inline");
    }

    #[test]
    fn normalise_rejects_duplicates() {
        let mut settings = Settings::default();
        settings.models.push(ModelProfile::builtin_deepseek());
        assert!(settings.normalise().is_err());
    }

    #[test]
    fn normalise_selects_first_profile_when_current_is_unknown() {
        let mut settings = Settings {
            current_model: Some("missing".to_string()),
            ..Default::default()
        };
        settings.normalise().unwrap();
        assert_eq!(settings.current_model.as_deref(), Some(DEFAULT_PROFILE));
    }

    #[test]
    fn unknown_profile_is_reported_with_available_names() {
        let settings = Settings::default();
        let error = settings.profile(Some("nope")).unwrap_err().to_string();
        assert!(error.contains("nope"));
        assert!(error.contains(DEFAULT_PROFILE));
    }
}
