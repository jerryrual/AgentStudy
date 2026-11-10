//! Typed failures at the ACP request and JSON-RPC boundary.

use std::error::Error as StdError;
use std::fmt;

use agent_client_protocol::v1::schema::{self, SessionId};
use agent_client_protocol::Error as ProtocolError;
use anyhow::Error;

/// A client-controlled session request field that failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvalidSessionRequest {
    AdditionalDirectories,
    McpServers,
    InvalidMcpServers,
    CwdNotAbsolute,
    CwdNotUtf8,
    CwdContainsNul,
    CwdMissing,
    CwdNotDirectory,
    CwdMismatch,
    InvalidSessionId,
    AlreadyRegistered,
    ActiveElsewhere,
    PromptLoading,
    PromptActive,
}

impl InvalidSessionRequest {
    fn message(self) -> &'static str {
        match self {
            Self::AdditionalDirectories => "additional workspace roots are not supported",
            Self::McpServers => "MCP servers are not supported for loaded sessions",
            Self::InvalidMcpServers => "invalid MCP server configuration",
            Self::CwdNotAbsolute => "cwd must be absolute",
            Self::CwdNotUtf8 => "cwd must be valid UTF-8",
            Self::CwdContainsNul => "cwd must not contain NUL",
            Self::CwdMissing => "cwd does not exist",
            Self::CwdNotDirectory => "cwd is not a directory",
            Self::CwdMismatch => "cwd does not match the persisted session",
            Self::InvalidSessionId => "session ID is invalid",
            Self::AlreadyRegistered => "session is already active",
            Self::ActiveElsewhere => "session is active in another process",
            Self::PromptLoading => "session is still loading",
            Self::PromptActive => "session already has an active prompt",
        }
    }
}

/// A classified ACP control failure whose internal source is never sent to clients.
#[derive(Debug)]
pub(crate) enum ControlError {
    Invalid(InvalidSessionRequest),
    SessionNotFound(SessionId),
    Internal(Error),
    Protocol(ProtocolError),
}

/// Result returned by ACP control boundaries before wire conversion.
pub(crate) type ControlResult<T> = Result<T, ControlError>;

impl ControlError {
    /// Classify one fixed client-controlled validation failure.
    pub(crate) fn invalid(reason: InvalidSessionRequest) -> Self {
        Self::Invalid(reason)
    }

    /// Preserve an internal source for local diagnostics and sanitized mapping.
    pub(crate) fn internal(error: impl Into<Error>) -> Self {
        Self::Internal(error.into())
    }
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid ACP session request: {}", reason.message()),
            Self::SessionNotFound(_) => f.write_str("ACP session not found"),
            Self::Internal(_) => f.write_str("internal ACP session error"),
            Self::Protocol(error) => write!(f, "ACP protocol error: {error}"),
        }
    }
}

impl StdError for ControlError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Internal(error) => Some(error.as_ref()),
            Self::Protocol(error) => Some(error),
            Self::Invalid(_) | Self::SessionNotFound(_) => None,
        }
    }
}

impl From<ControlError> for ProtocolError {
    fn from(error: ControlError) -> Self {
        match error {
            ControlError::Invalid(reason) => ProtocolError::invalid_params().data(reason.message()),
            ControlError::SessionNotFound(session_id) => {
                schema::Error::session_not_found(session_id).into()
            }
            ControlError::Internal(error) => {
                eprintln!("warning: ACP internal error: {error:#}");
                ProtocolError::internal_error()
            }
            ControlError::Protocol(error) => error,
        }
    }
}

#[cfg(test)]
mod tests {
    use agent_client_protocol::ErrorCode;

    use super::*;

    #[test]
    fn maps_invalid_reasons_to_fixed_safe_protocol_data() {
        let cases = [
            (
                InvalidSessionRequest::AdditionalDirectories,
                "additional workspace roots are not supported",
            ),
            (
                InvalidSessionRequest::McpServers,
                "MCP servers are not supported for loaded sessions",
            ),
            (
                InvalidSessionRequest::InvalidMcpServers,
                "invalid MCP server configuration",
            ),
            (
                InvalidSessionRequest::CwdNotAbsolute,
                "cwd must be absolute",
            ),
            (InvalidSessionRequest::CwdNotUtf8, "cwd must be valid UTF-8"),
            (
                InvalidSessionRequest::CwdContainsNul,
                "cwd must not contain NUL",
            ),
            (InvalidSessionRequest::CwdMissing, "cwd does not exist"),
            (
                InvalidSessionRequest::CwdNotDirectory,
                "cwd is not a directory",
            ),
            (
                InvalidSessionRequest::CwdMismatch,
                "cwd does not match the persisted session",
            ),
            (
                InvalidSessionRequest::InvalidSessionId,
                "session ID is invalid",
            ),
            (
                InvalidSessionRequest::AlreadyRegistered,
                "session is already active",
            ),
            (
                InvalidSessionRequest::ActiveElsewhere,
                "session is active in another process",
            ),
            (
                InvalidSessionRequest::PromptLoading,
                "session is still loading",
            ),
            (
                InvalidSessionRequest::PromptActive,
                "session already has an active prompt",
            ),
        ];

        for (reason, message) in cases {
            let error: ProtocolError = ControlError::Invalid(reason).into();
            assert_eq!(error.code, ErrorCode::INVALID_PARAMS);
            assert_eq!(error.data, Some(serde_json::json!(message)));
        }
    }

    #[test]
    fn maps_missing_sessions_with_the_requested_typed_id() {
        let error: ProtocolError =
            ControlError::SessionNotFound(SessionId::new("missing-session")).into();

        assert_eq!(error.code, ErrorCode::RESOURCE_NOT_FOUND);
        assert_eq!(
            error.data,
            Some(serde_json::json!({"sessionId": "missing-session"}))
        );
    }

    #[test]
    fn hides_internal_sources_but_preserves_protocol_errors() {
        let internal: ProtocolError =
            ControlError::internal(anyhow::anyhow!("SECRET_CANARY/session.jsonl")).into();
        assert_eq!(internal.code, ErrorCode::INTERNAL_ERROR);
        assert_eq!(internal.data, None);
        assert!(!serde_json::to_string(&internal)
            .unwrap()
            .contains("SECRET_CANARY"));

        let protocol = ProtocolError::request_cancelled().data("transport detail");
        let mapped: ProtocolError = ControlError::Protocol(protocol.clone()).into();
        assert_eq!(mapped, protocol);
    }
}
