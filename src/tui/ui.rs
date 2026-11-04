//! Rendering of the iCode TUI.

use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::app::{approval_layout, App, Modal, ToolStatus, ViewItem};
use crate::markdown;
use crate::theme::Theme;

const SPINNER: [&str; 4] = ["|", "/", "-", "\\"];

/// Draw the whole application into `frame`.
pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let layout = Layout::vertical([
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .split(area);
    draw_transcript(frame, app, layout[0]);
    draw_prompt(frame, app, layout[1]);
    draw_status(frame, app, layout[2]);
    match &app.modal {
        Some(Modal::Approval { request, .. }) => {
            draw_approval(frame, area, app, &request.tool, &request.summary)
        }
        Some(Modal::Help) => draw_help(frame, area, app),
        None => {}
    }
    if app.modal.is_none() {
        set_prompt_cursor(frame, app, layout[1]);
    }
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let lines = build_transcript(app);
    let width = area.width as usize;
    let lines: Vec<Line<'static>> = lines
        .into_iter()
        .flat_map(|line| wrap_line(line, width))
        .collect();
    let total = lines.len();
    let viewport = area.height as usize;
    let offset = if app.follow {
        total.saturating_sub(viewport)
    } else {
        total
            .saturating_sub(viewport)
            .saturating_sub(app.scroll_back as usize)
    };
    let offset = offset.min(u16::MAX as usize) as u16;
    frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), area);
}

fn build_transcript(app: &App) -> Vec<Line<'static>> {
    let theme = &app.theme;
    let mut lines: Vec<Line<'static>> = Vec::new();
    for item in &app.view {
        match item {
            ViewItem::User(text) => {
                push_prefixed(
                    &mut lines,
                    "❯ ",
                    &expand_tabs(text),
                    Style::default().fg(theme.user),
                );
                lines.push(Line::default());
            }
            ViewItem::Assistant {
                text,
                reasoning,
                streaming,
            } => {
                if !reasoning.is_empty() {
                    let preview: Vec<&str> = reasoning.lines().rev().take(3).collect();
                    for line in preview.into_iter().rev() {
                        lines.push(Line::from(Span::styled(
                            format!("  · {line}"),
                            Style::default()
                                .fg(theme.dim)
                                .add_modifier(Modifier::ITALIC),
                        )));
                    }
                }
                let rendered = markdown::render(&expand_tabs(text), theme);
                if rendered.is_empty() && *streaming {
                    lines.push(Line::from(Span::styled(
                        format!("  {} thinking…", SPINNER[app.spinner % SPINNER.len()]),
                        Style::default().fg(theme.dim),
                    )));
                } else {
                    lines.extend(rendered);
                    if *streaming {
                        append_cursor_marker(&mut lines, theme);
                    }
                }
                if !*streaming {
                    lines.push(Line::default());
                }
            }
            ViewItem::Tool {
                name,
                summary,
                status,
                ok,
                duration,
                output,
            } => {
                let status_span = match status {
                    ToolStatus::Running => Span::styled(
                        format!(" {} running", SPINNER[app.spinner % SPINNER.len()]),
                        Style::default().fg(theme.warning),
                    ),
                    ToolStatus::Done => {
                        let (mark, color) = match ok {
                            Some(true) => ("✓", theme.success),
                            _ => ("✗", theme.error),
                        };
                        let duration = duration
                            .map(|duration| format!(" {:.1}s", duration.as_secs_f64()))
                            .unwrap_or_default();
                        Span::styled(format!(" {mark}{duration}"), Style::default().fg(color))
                    }
                };
                let mut header = vec![
                    Span::styled("→ ", Style::default().fg(theme.dim)),
                    Span::styled(name.clone(), Style::default().fg(theme.tool)),
                ];
                if !summary.is_empty() {
                    header.push(Span::styled(
                        format!("  {summary}"),
                        Style::default().fg(theme.dim),
                    ));
                }
                header.push(status_span);
                lines.push(Line::from(header));
                for line in expand_tabs(output).lines() {
                    lines.push(Line::from(Span::styled(
                        format!("  {line}"),
                        Style::default().fg(theme.dim),
                    )));
                }
            }
            ViewItem::Info(text) => {
                if text.is_empty() {
                    lines.push(Line::default());
                } else {
                    for line in expand_tabs(text).lines() {
                        lines.push(Line::from(Span::styled(
                            format!("  {line}"),
                            Style::default().fg(theme.dim),
                        )));
                    }
                }
            }
            ViewItem::Error(text) => {
                for line in text.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("✗ {line}"),
                        Style::default().fg(theme.error),
                    )));
                }
            }
        }
    }
    lines
}

fn expand_tabs(text: &str) -> String {
    if text.contains('\t') {
        text.replace('\t', "    ")
    } else {
        text.to_string()
    }
}

fn push_prefixed(lines: &mut Vec<Line<'static>>, prefix: &str, text: &str, style: Style) {
    let mut first = true;
    for line in text.split('\n') {
        if first {
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), style.add_modifier(Modifier::BOLD)),
                Span::styled(line.to_string(), style),
            ]));
            first = false;
        } else {
            lines.push(Line::from(Span::styled(
                format!("{} {line}", " ".repeat(prefix.width())),
                style,
            )));
        }
    }
    if first {
        lines.push(Line::from(Span::styled(prefix.to_string(), style)));
    }
}

/// Character-wrap styled lines to `width` columns.
fn wrap_line(line: Line<'static>, width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return vec![line];
    }
    let mut result = Vec::new();
    let mut current = Line::default();
    let mut current_width = 0usize;
    for span in line.spans {
        let style = span.style;
        let mut chunk = String::new();
        for character in span.content.chars() {
            let character_width = char_width(character);
            if current_width + character_width > width && current_width > 0 {
                if !chunk.is_empty() {
                    current
                        .spans
                        .push(Span::styled(std::mem::take(&mut chunk), style));
                }
                result.push(std::mem::take(&mut current));
                current_width = 0;
            }
            chunk.push(character);
            current_width += character_width;
            if current_width >= width {
                current
                    .spans
                    .push(Span::styled(std::mem::take(&mut chunk), style));
                result.push(std::mem::take(&mut current));
                current_width = 0;
            }
        }
        if !chunk.is_empty() {
            current.spans.push(Span::styled(chunk, style));
        }
    }
    if !current.spans.is_empty() || result.is_empty() {
        result.push(current);
    }
    result
}

fn append_cursor_marker(lines: &mut Vec<Line<'static>>, theme: &Theme) {
    if let Some(last) = lines.last_mut() {
        if last.spans.is_empty() {
            *last = Line::from(Span::styled("▍", Style::default().fg(theme.accent)));
        } else {
            last.spans
                .push(Span::styled("▍", Style::default().fg(theme.accent)));
        }
    } else {
        lines.push(Line::from(Span::styled(
            "▍",
            Style::default().fg(theme.accent),
        )));
    }
}

fn draw_prompt(frame: &mut Frame, app: &App, area: Rect) {
    let border_color = if app.pending_ask.is_some() {
        app.theme.warning
    } else if app.running {
        app.theme.dim
    } else {
        app.theme.border
    };
    let title = if app.pending_ask.is_some() {
        " answer "
    } else {
        " prompt "
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .title(Span::styled(title, Style::default().fg(app.theme.dim)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let (visible, _) = visible_input(app, inner.width.saturating_sub(2) as usize);
    let mut spans = vec![Span::styled("❯ ", Style::default().fg(app.theme.accent))];
    if visible.is_empty() {
        spans.push(Span::styled(
            "Ask a coding question, or type /help",
            Style::default().fg(app.theme.dim),
        ));
    } else {
        spans.push(Span::styled(visible, Style::default().fg(app.theme.user)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), inner);
}

/// Clip the input from the left so the cursor stays visible.
fn visible_input(app: &App, available: usize) -> (String, usize) {
    let cursor_bytes = app
        .input
        .char_indices()
        .nth(app.cursor)
        .map(|(byte, _)| byte)
        .unwrap_or(app.input.len());
    let (before_cursor, _after_cursor) = app.input.split_at(cursor_bytes);
    let before_width = before_cursor.width();
    let mut start = 0usize;
    let mut visible_before = before_width;
    if available > 0 && visible_before >= available {
        // Drop characters from the start until the cursor fits.
        let mut drop_width = visible_before - available + 1;
        for (index, character) in before_cursor.char_indices() {
            if drop_width == 0 {
                break;
            }
            drop_width = drop_width.saturating_sub(char_width(character));
            start = index + character.len_utf8();
        }
        visible_before = app.input[start..cursor_bytes].width();
    }
    let mut visible = String::new();
    let mut width = 0usize;
    for (index, character) in app.input.char_indices() {
        if index < start {
            continue;
        }
        let character_width = char_width(character);
        if width + character_width > available && available > 0 {
            break;
        }
        visible.push(character);
        width += character_width;
    }
    (visible, visible_before)
}

fn set_prompt_cursor(frame: &mut Frame, app: &App, area: Rect) {
    let inner = Block::bordered().inner(area);
    let available = inner.width.saturating_sub(2) as usize;
    let (_, before_width) = visible_input(app, available);
    let x = inner
        .x
        .saturating_add(2)
        .saturating_add(before_width as u16);
    let y = inner.y;
    if inner.width > 2 {
        frame.set_cursor_position(ratatui::layout::Position::new(
            x.min(inner.right().saturating_sub(1)),
            y,
        ));
    }
}

fn char_width(character: char) -> usize {
    UnicodeWidthChar::width(character).unwrap_or(0).max(1)
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![Span::styled(
        app.status_line(),
        Style::default().fg(app.theme.dim),
    )];
    if app.running {
        spans.insert(
            0,
            Span::styled(
                format!("{} ", SPINNER[app.spinner % SPINNER.len()]),
                Style::default().fg(app.theme.warning),
            ),
        );
    }
    let line = Line::from(spans);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_approval(frame: &mut Frame, area: Rect, app: &App, tool: &str, summary: &str) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(app.theme.warning))
        .title(Span::styled(
            " Approval required ",
            Style::default()
                .fg(app.theme.warning)
                .add_modifier(Modifier::BOLD),
        ));
    // Grow the dialog so the complete command stays reviewable. The summary is
    // wrapped here (not by Paragraph) because commands are often a single long
    // word that Paragraph does not break.
    let layout = approval_layout(area, tool, summary);
    let interior = layout.interior;
    let clipped = layout.clipped;
    let mut lines = Vec::new();
    if clipped {
        lines.extend(wrap_line(
            Line::from(Span::styled(
                "  ⚠ command clipped — approval disabled; resize, or press n/esc",
                Style::default().fg(app.theme.warning),
            )),
            interior,
        ));
    }
    lines.extend(wrap_line(
        Line::from(Span::styled(
            format!("  {tool}: {summary}"),
            Style::default().fg(app.theme.text),
        )),
        interior,
    ));
    if !clipped {
        lines.push(Line::default());
        lines.extend(wrap_line(
            Line::from(Span::styled(
                "  y allow once · a always allow · n deny · esc deny",
                Style::default().fg(app.theme.dim),
            )),
            interior,
        ));
    }
    let height = (lines.len() as u16 + 2)
        .clamp(7, area.height.max(7))
        .min(area.height.max(1));
    let popup = centered(area, 70, height);
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

fn draw_help(frame: &mut Frame, area: Rect, app: &App) {
    let entries = [
        ("/help", "show this help"),
        ("/new, /clear", "start a new session"),
        ("/models, /model <name>", "list or switch model profiles"),
        ("/approval ask|auto|deny", "change the approval mode"),
        ("/sessions, /resume <id>", "list or resume sessions"),
        ("/compact", "summarize the conversation to free context"),
        ("/cwd <path>", "change the working directory"),
        ("!<command>", "run a shell command"),
        ("@<file>", "attach a file to the message"),
        ("esc", "cancel a running turn"),
        ("ctrl-c", "quit"),
    ];
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "Commands and shortcuts",
        Style::default()
            .fg(app.theme.accent)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::default());
    for (command, description) in entries {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {command:<24}"),
                Style::default().fg(app.theme.accent),
            ),
            Span::styled(description.to_string(), Style::default().fg(app.theme.dim)),
        ]));
    }
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        "  Esc closes this panel.",
        Style::default().fg(app.theme.dim),
    )));
    let popup = centered(area, 70, 17);
    frame.render_widget(Clear, popup);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(app.theme.border))
        .title(Span::styled(" help ", Style::default().fg(app.theme.dim)));
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

fn centered(area: Rect, percent_x: u16, height: u16) -> Rect {
    let vertical = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .split(area);
    let horizontal = Layout::horizontal([Constraint::Percentage(percent_x)])
        .flex(Flex::Center)
        .split(vertical[0]);
    horizontal[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentKind;
    use crate::config::{ApprovalMode, ModelProfile, Settings};
    use crate::session::SessionStore;
    use crate::tui::app::{App, AppOptions};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::path::PathBuf;

    fn test_app() -> App {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().to_path_buf();
        // Keep the temp dir alive for the lifetime of the test process section.
        std::mem::forget(directory);
        let mut profile = ModelProfile::builtin_deepseek();
        profile.api_key = "test-key".to_string();
        profile.base_url = "http://127.0.0.1:1".to_string();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let handle = runtime.handle().clone();
        std::mem::forget(runtime);
        App::new(
            handle,
            AppOptions {
                settings: Settings::default(),
                profile,
                agent: AgentKind::Code,
                cwd: PathBuf::from("/tmp/project"),
                store: SessionStore::with_root(path.join("sessions")),
                resume: None,
                theme: Theme::default(),
                approval: ApprovalMode::Ask,
            },
        )
        .unwrap()
    }

    fn render(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let mut output = String::new();
        for row in buffer.area.rows() {
            let mut line = String::new();
            for x in row.x..row.x + row.width {
                line.push_str(buffer[(x, row.y)].symbol());
            }
            output.push_str(line.trim_end());
            output.push('\n');
        }
        output
    }

    #[test]
    fn renders_welcome_status_and_prompt() {
        let mut app = test_app();
        let output = render(&mut app, 100, 24);
        assert!(output.contains("Welcome to icode"), "{output}");
        assert!(output.contains("Code Agent"), "{output}");
        assert!(output.contains("prompt"), "{output}");
        assert!(output.contains("Ask a coding question"), "{output}");
    }

    #[test]
    fn renders_user_message_and_markdown_answer() {
        let mut app = test_app();
        app.view.push(ViewItem::User("add a test".to_string()));
        app.view.push(ViewItem::Assistant {
            text: "Sure — here is **the plan**:\n\n- read the file".to_string(),
            reasoning: String::new(),
            streaming: false,
        });
        let output = render(&mut app, 100, 30);
        assert!(output.contains("❯ add a test"), "{output}");
        assert!(output.contains("the plan"), "{output}");
        assert!(output.contains("read the file"), "{output}");
    }

    #[test]
    fn renders_tool_blocks() {
        let mut app = test_app();
        app.view.push(ViewItem::Tool {
            name: "bash".to_string(),
            summary: "cargo test".to_string(),
            status: ToolStatus::Done,
            ok: Some(true),
            duration: Some(std::time::Duration::from_millis(1200)),
            output: "exit code: 0".to_string(),
        });
        let output = render(&mut app, 100, 24);
        assert!(output.contains("→ bash"), "{output}");
        assert!(output.contains("cargo test"), "{output}");
        assert!(output.contains("1.2s"), "{output}");
        assert!(output.contains("exit code: 0"), "{output}");
    }

    #[test]
    fn duplicate_parallel_tools_keep_outputs_in_call_order() {
        use crate::tui::app::UiEvent;

        let mut app = test_app();
        app.running = true;
        for _ in 0..2 {
            app.push_event(UiEvent::ToolStarted {
                name: "read_file".to_string(),
                summary: "same.txt".to_string(),
            });
        }
        for output in ["first result", "second result"] {
            app.push_event(UiEvent::ToolFinished {
                name: "read_file".to_string(),
                ok: true,
                duration: std::time::Duration::from_millis(1),
                summary: "same.txt".to_string(),
                output: output.to_string(),
            });
        }
        app.pump();

        let outputs: Vec<_> = app
            .view
            .iter()
            .filter_map(|item| match item {
                ViewItem::Tool { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs, ["first result", "second result"]);
    }

    #[test]
    fn cancellation_closes_running_tool_cards() {
        let mut app = test_app();
        app.running = true;
        app.view.push(ViewItem::Tool {
            name: "read_file".to_string(),
            summary: "same.txt".to_string(),
            status: ToolStatus::Running,
            ok: None,
            duration: None,
            output: String::new(),
        });

        app.cancel();

        match app
            .view
            .iter()
            .find(|item| matches!(item, ViewItem::Tool { .. }))
        {
            Some(ViewItem::Tool {
                status, ok, output, ..
            }) => {
                assert_eq!(*status, ToolStatus::Done);
                assert_eq!(*ok, Some(false));
                assert_eq!(output, "Cancelled.");
            }
            _ => panic!("expected tool card"),
        }
    }

    #[test]
    fn cancellation_ignores_queued_tool_events() {
        use crate::tui::app::UiEvent;

        let mut app = test_app();
        app.running = true;
        app.push_event(UiEvent::ToolStarted {
            name: "read_file".to_string(),
            summary: "queued.txt".to_string(),
        });
        app.push_event(UiEvent::ToolFinished {
            name: "read_file".to_string(),
            ok: true,
            duration: std::time::Duration::from_millis(1),
            summary: "queued.txt".to_string(),
            output: "late result".to_string(),
        });

        app.cancel();
        app.pump();

        assert!(
            !app.view.iter().any(
                |item| matches!(item, ViewItem::Tool { summary, .. } if summary == "queued.txt")
            ),
            "cancelled turn must not recreate a tool card"
        );
    }

    #[test]
    fn stale_turn_events_cannot_mutate_a_new_turn() {
        use crate::tui::app::UiEvent;

        let mut app = test_app();
        app.running = true;
        app.push_stale_event(UiEvent::ToolFinished {
            name: "bash".to_string(),
            ok: true,
            duration: std::time::Duration::from_millis(1),
            summary: "new command".to_string(),
            output: "old result".to_string(),
        });
        app.push_stale_event(UiEvent::ShellFinished {
            ok: true,
            duration: std::time::Duration::from_millis(1),
        });
        app.pump();

        assert!(app.running, "stale ShellFinished must not end the new turn");
        assert!(
            !app.view.iter().any(
                |item| matches!(item, ViewItem::Tool { output, .. } if output == "old result")
            ),
            "stale ToolFinished must not create or finish a card"
        );
    }

    #[test]
    fn compaction_uses_persisted_messages_after_cancelled_turn() {
        use crate::model::{AssistantMessage, Message};

        let mut app = test_app();
        app.messages = vec![Message::user("old request")];
        app.session
            .append_message(&Message::user("old request"))
            .unwrap();
        app.session
            .append_message(&Message::assistant(AssistantMessage::text(
                "persisted result from a completed round",
            )))
            .unwrap();

        let messages = app.compaction_messages().unwrap();

        assert_eq!(messages.len(), 2);
        assert!(messages[1].preview().contains("persisted result"));
        assert_eq!(app.messages.len(), 1, "memory intentionally remains stale");
    }

    #[test]
    fn completed_compaction_replaces_and_persists_model_context() {
        use crate::compaction::Compaction;
        use crate::model::{Message, Usage};
        use crate::tui::app::UiEvent;

        let mut app = test_app();
        app.messages = vec![Message::user("old request")];
        app.session
            .append_message(&Message::user("old request"))
            .unwrap();
        app.running = true;
        app.push_event(UiEvent::CompactionFinished {
            result: Ok(Compaction {
                summary: "Changed src/main.rs; tests pass.".to_string(),
                usage: Usage {
                    prompt_tokens: 20,
                    completion_tokens: 8,
                    total_tokens: 28,
                },
            }),
            duration: std::time::Duration::from_millis(5),
        });
        app.pump();

        assert!(!app.running);
        assert_eq!(app.messages.len(), 1);
        assert!(app.messages[0]
            .preview()
            .contains("Earlier conversation summary"));
        assert!(app.view.iter().any(
            |item| matches!(item, ViewItem::Assistant { text, .. } if text.contains("tests pass"))
        ));
        let resumed = app.store.load(app.session.id()).unwrap().messages();
        assert_eq!(resumed, app.messages);
    }

    #[test]
    fn help_lists_compaction_without_clipping_the_close_hint() {
        let mut app = test_app();
        app.modal = Some(Modal::Help);

        let output = render(&mut app, 100, 24);

        assert!(output.contains("/compact"), "{output}");
        assert!(output.contains("Esc closes this panel."), "{output}");
    }

    #[test]
    fn renders_approval_modal() {
        let mut app = test_app();
        let (reply, _rx) = tokio::sync::oneshot::channel();
        app.modal = Some(Modal::Approval {
            request: crate::interaction::ApprovalRequest::new("bash", "cargo test"),
            reply,
        });
        let output = render(&mut app, 100, 24);
        assert!(output.contains("Approval required"), "{output}");
        assert!(output.contains("y allow once"), "{output}");
    }

    #[test]
    fn approval_modal_wraps_long_commands() {
        let mut app = test_app();
        let (reply, _rx) = tokio::sync::oneshot::channel();
        let command = format!("echo {} THE-END-MARKER", "x".repeat(240));
        app.modal = Some(Modal::Approval {
            request: crate::interaction::ApprovalRequest::new("bash", &command),
            reply,
        });
        let output = render(&mut app, 100, 30);
        assert!(output.contains("THE-END-MARKER"), "{output}");
    }

    #[test]
    fn clipped_command_disables_approval() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut app = test_app();
        let (reply, mut rx) = tokio::sync::oneshot::channel();
        let command = format!("echo {}", "x".repeat(400));
        app.modal = Some(Modal::Approval {
            request: crate::interaction::ApprovalRequest::new("bash", &command),
            reply,
        });
        let screen = ratatui::layout::Rect::new(0, 0, 60, 9);
        app.handle_key(
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
            screen,
        );
        assert!(app.modal.is_some(), "clipped approval stays open");
        assert!(rx.try_recv().is_err(), "no decision is sent");

        let output = render(&mut app, 60, 9);
        assert!(output.contains("approval disabled"), "{output}");
    }

    #[test]
    fn enter_does_not_approve_a_tool_call() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut app = test_app();
        let (reply, mut rx) = tokio::sync::oneshot::channel();
        app.modal = Some(Modal::Approval {
            request: crate::interaction::ApprovalRequest::new("bash", "cargo test"),
            reply,
        });
        let screen = ratatui::layout::Rect::new(0, 0, 100, 30);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), screen);
        assert!(app.modal.is_some());
        assert!(rx.try_recv().is_err());

        app.handle_key(
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
            screen,
        );
        assert!(app.modal.is_none());
        assert_eq!(
            rx.try_recv().unwrap(),
            crate::interaction::ApprovalDecision::AllowOnce
        );
    }

    #[test]
    fn wrapping_an_exact_width_line_adds_no_blank_line() {
        let line = Line::from(Span::raw("abcd"));
        assert_eq!(wrap_line(line, 4).len(), 1);
        let line = Line::from(Span::raw(""));
        assert_eq!(wrap_line(line, 4).len(), 1);
    }

    #[test]
    fn input_cursor_is_visible_when_wrapping() {
        let mut app = test_app();
        app.input = "x".repeat(200);
        app.cursor = 200;
        let output = render(&mut app, 60, 16);
        assert!(output.contains("xxxx"), "{output}");
    }
}
