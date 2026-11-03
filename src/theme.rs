//! Terminal colour theme.
//!
//! iCode ships a set of themes edited through the TUI. The Rust port keeps a
//! single, terminal-friendly palette for now; the type exists so that themes
//! can be added without touching the renderer.

use ratatui::style::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    /// Brand and emphasis colour (headings, bullets, links).
    pub accent: Color,
    /// Secondary text: metadata, hints and reasoning.
    pub dim: Color,
    /// Errors and denied operations.
    pub error: Color,
    /// Successful tool calls and completions.
    pub success: Color,
    /// Warnings and in-progress indicators.
    pub warning: Color,
    /// User text.
    pub user: Color,
    /// Tool call headers.
    pub tool: Color,
    /// Inline code and code blocks.
    pub code: Color,
    /// Border colour for the prompt box.
    pub border: Color,
    /// Default text colour (terminal foreground).
    pub text: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            accent: Color::Cyan,
            dim: Color::DarkGray,
            error: Color::Red,
            success: Color::Green,
            warning: Color::Yellow,
            user: Color::Blue,
            tool: Color::Magenta,
            code: Color::Yellow,
            border: Color::DarkGray,
            text: Color::Reset,
        }
    }
}
