//! A small Markdown renderer for the terminal.
//!
//! iCode renders assistant answers as rich Markdown — headings, lists, inline
//! code and fenced code blocks. This renderer covers that core subset without a
//! heavy dependency; the output is [`ratatui::text::Line`]s ready to be wrapped
//! by a `Paragraph`.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::Theme;

/// Render Markdown text into styled lines.
pub fn render(text: &str, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    // Length of the opening backtick fence, if a code block is open.
    let mut fence: Option<usize> = None;
    for raw in text.split('\n') {
        let trimmed = raw.trim_start();
        let ticks = trimmed
            .chars()
            .take_while(|character| *character == '`')
            .count();
        if ticks >= 3 {
            if let Some(open) = fence {
                if ticks >= open && trimmed[ticks..].trim().is_empty() {
                    fence = None;
                    continue;
                }
                // A shorter or annotated fence inside a block is content.
                lines.push(Line::from(Span::styled(
                    format!("  │ {raw}"),
                    Style::default().fg(theme.code),
                )));
                continue;
            }
            fence = Some(ticks);
            let language = trimmed.trim_start_matches('`').trim();
            let label = if language.is_empty() {
                "code".to_string()
            } else {
                language.to_string()
            };
            lines.push(Line::from(Span::styled(
                format!("  ┌─ {label}"),
                Style::default().fg(theme.dim),
            )));
            continue;
        }
        if fence.is_some() {
            lines.push(Line::from(Span::styled(
                format!("  │ {raw}"),
                Style::default().fg(theme.code),
            )));
            continue;
        }
        if trimmed.is_empty() {
            lines.push(Line::default());
            continue;
        }
        if let Some(heading) = heading_text(trimmed) {
            lines.push(Line::from(Span::styled(
                heading.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(item) = bullet_text(trimmed) {
            let mut spans = vec![Span::styled("• ", Style::default().fg(theme.accent))];
            spans.extend(inline_spans(item, theme));
            lines.push(Line::from(spans));
            continue;
        }
        if let Some((marker, item)) = ordered_item(trimmed) {
            let mut spans = vec![Span::styled(
                format!("{marker} "),
                Style::default().fg(theme.accent),
            )];
            spans.extend(inline_spans(item, theme));
            lines.push(Line::from(spans));
            continue;
        }
        if let Some(quote) = trimmed.strip_prefix("> ") {
            let mut spans = vec![Span::styled("│ ", Style::default().fg(theme.dim))];
            spans.extend(inline_spans(quote, theme));
            let style = Style::default().fg(theme.dim);
            lines.push(Line::from(
                spans
                    .into_iter()
                    .map(|span| Span::styled(span.content.into_owned(), span.style.patch(style)))
                    .collect::<Vec<_>>(),
            ));
            continue;
        }
        lines.push(Line::from(inline_spans(raw, theme)));
    }
    lines
}

fn heading_text(trimmed: &str) -> Option<&str> {
    let hashes = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = trimmed[hashes..].trim_start();
    if !trimmed[hashes..].starts_with(' ') {
        return None;
    }
    Some(rest)
}

fn bullet_text(trimmed: &str) -> Option<&str> {
    trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
}

fn ordered_item(trimmed: &str) -> Option<(&str, &str)> {
    let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let rest = &trimmed[digits..];
    let rest = rest
        .strip_prefix(". ")
        .or_else(|| rest.strip_prefix(") "))?;
    Some((&trimmed[..digits], rest))
}

/// Parse inline Markdown: `code`, **bold**, *italic* and [links](url).
fn inline_spans(text: &str, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('`') {
            if let Some(end) = after.find('`') {
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    after[..end].to_string(),
                    Style::default().fg(theme.code),
                ));
                rest = &after[end + 1..];
                continue;
            }
        }
        if let Some(after) = rest.strip_prefix("**") {
            if let Some(end) = after.find("**") {
                flush_plain(&mut plain, &mut spans);
                spans.push(Span::styled(
                    after[..end].to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                rest = &after[end + 2..];
                continue;
            }
        }
        if let Some(after) = rest.strip_prefix('*') {
            if let Some(end) = after.find('*') {
                let content = &after[..end];
                if !content.is_empty() && !content.contains('\n') {
                    flush_plain(&mut plain, &mut spans);
                    spans.push(Span::styled(
                        content.to_string(),
                        Style::default().add_modifier(Modifier::ITALIC),
                    ));
                    rest = &after[end + 1..];
                    continue;
                }
            }
        }
        if let Some(after) = rest.strip_prefix('[') {
            if let Some(label_end) = after.find("](") {
                if let Some(url_end) = after[label_end + 2..].find(')') {
                    let label = &after[..label_end];
                    let url = &after[label_end + 2..label_end + 2 + url_end];
                    flush_plain(&mut plain, &mut spans);
                    spans.push(Span::styled(
                        label.to_string(),
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::UNDERLINED),
                    ));
                    if url != label {
                        spans.push(Span::styled(
                            format!(" ({url})"),
                            Style::default().fg(theme.dim),
                        ));
                    }
                    rest = &after[label_end + 2 + url_end + 1..];
                    continue;
                }
            }
        }
        let character = rest.chars().next().unwrap_or_default();
        plain.push(character);
        rest = &rest[character.len_utf8()..];
    }
    flush_plain(&mut plain, &mut spans);
    spans
}

fn flush_plain(plain: &mut String, spans: &mut Vec<Span<'static>>) {
    if !plain.is_empty() {
        spans.push(Span::raw(std::mem::take(plain)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Color;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    #[test]
    fn renders_headings_bullets_and_code_blocks() {
        let theme = Theme::default();
        let lines = render("# Title\n\n- first\n\n```rust\nlet x = 1;\n```", &theme);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(text[0], "Title");
        assert!(text.iter().any(|line| line == "• first"));
        assert!(text.iter().any(|line| line.contains("┌─ rust")));
        assert!(text.iter().any(|line| line.contains("let x = 1;")));
        assert_eq!(lines[0].spans[0].style.fg, Some(Color::Cyan));
    }

    #[test]
    fn renders_inline_styles() {
        let theme = Theme::default();
        let lines = render(
            "use `cargo test` and **bold** [docs](https://example.com)",
            &theme,
        );
        let spans = &lines[0].spans;
        assert!(spans.iter().any(|span| span.content == "cargo test"));
        assert!(spans
            .iter()
            .any(|span| span.content == "bold" && span.style.add_modifier == Modifier::BOLD));
        assert!(spans.iter().any(|span| span.content == "docs"));
        assert!(spans
            .iter()
            .any(|span| span.content.contains("https://example.com")));
    }

    #[test]
    fn keeps_ordered_lists() {
        let theme = Theme::default();
        let lines = render("1. one\n2) two", &theme);
        assert_eq!(line_text(&lines[0]), "1 one");
        assert_eq!(line_text(&lines[1]), "2 two");
    }

    #[test]
    fn unterminated_code_fence_is_still_rendered() {
        let theme = Theme::default();
        let lines = render("```\nnot closed", &theme);
        assert!(line_text(&lines[0]).contains("code"));
        assert!(line_text(&lines[1]).contains("not closed"));
    }

    #[test]
    fn closes_a_block_only_on_a_matching_fence() {
        let theme = Theme::default();
        let lines = render(
            "````markdown\n```rust\nlet x = 1;\n```\n````\n\ntext after",
            &theme,
        );
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert!(text.iter().any(|line| line.contains("```rust")), "{text:?}");
        assert!(
            text.iter().any(|line| line.contains("let x = 1;")),
            "{text:?}"
        );
        assert!(text.iter().any(|line| line == "text after"), "{text:?}");
    }
}
