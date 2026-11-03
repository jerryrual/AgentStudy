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
use serde::{Deserialize, Serialize};

/// Name of the built-in profile created on first run.
pub const DEFAULT_PROFILE: &str = "deepseek";

const DEFAULT_TIMEOUT_SECS: u64 = 30;

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
    /// The Responses API is accepted in configuration but not implemented yet.
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
    #[serde(default)]
    pub temperature: Option<f32>,
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
            temperature: Some(0.1),
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
    /// Maximum number of model calls in one user turn.
    #[serde(default = "default_max_rounds")]
    pub max_tool_rounds: usize,
}

fn default_agent() -> String {
    "code".to_string()
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
    let stdout_reader = std::thread::spawn(move || read_all(stdout));
    let stderr_reader = std::thread::spawn(move || read_all(stderr));

    let started = Instant::now();
    let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    bail!("API key command {command:?} timed out after {DEFAULT_TIMEOUT_SECS}s");
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                bail!("failed to wait for API key command {command:?}: {error}");
            }
        }
    };

    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr);
        bail!(
            "API key command {command:?} exited with {status}: {}",
            stderr.trim()
        );
    }
    let key = String::from_utf8_lossy(&stdout).trim().to_string();
    if key.is_empty() {
        bail!("API key command {command:?} produced no output");
    }
    Ok(key)
}

fn read_all(mut reader: impl std::io::Read) -> Vec<u8> {
    let mut buffer = Vec::new();
    let _ = reader.read_to_end(&mut buffer);
    buffer
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
