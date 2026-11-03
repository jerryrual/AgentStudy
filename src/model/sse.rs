//! Incremental Server-Sent-Events parser for the streaming API.
//!
//! The parser is deliberately independent from HTTP so it can be unit tested
//! with arbitrary chunk boundaries.

/// Largest pending line or accumulated event payload accepted from a provider.
pub const MAX_EVENT_BYTES: usize = 1024 * 1024;

/// The provider sent an unterminated line or event larger than [`MAX_EVENT_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SseTooLarge;

impl std::fmt::Display for SseTooLarge {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "streaming event exceeded the {MAX_EVENT_BYTES}-byte limit"
        )
    }
}

impl std::error::Error for SseTooLarge {}

/// Accumulates bytes and yields `data:` payloads of complete SSE events.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    data: String,
    has_data: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes; returns every complete event payload.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, SseTooLarge> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=position).collect();
            line.pop(); // '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line).to_string();
            self.consume_line(&line, &mut events);
        }
        if self.buffer.len() > MAX_EVENT_BYTES || self.data.len() > MAX_EVENT_BYTES {
            return Err(SseTooLarge);
        }
        Ok(events)
    }

    /// Flush the trailing event at end of stream, if the server omitted the
    /// final blank line.
    pub fn finish(&mut self) -> Result<Vec<String>, SseTooLarge> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            if self.buffer.len() > MAX_EVENT_BYTES {
                return Err(SseTooLarge);
            }
            let line = String::from_utf8_lossy(&self.buffer).to_string();
            self.buffer.clear();
            self.consume_line(&line, &mut events);
        }
        if self.has_data {
            events.push(std::mem::take(&mut self.data));
            self.has_data = false;
        }
        Ok(events)
    }

    fn consume_line(&mut self, line: &str, events: &mut Vec<String>) {
        if line.is_empty() {
            if self.has_data {
                events.push(std::mem::take(&mut self.data));
                self.has_data = false;
            }
            return;
        }
        if line.starts_with(':') {
            // Comment / keep-alive.
            return;
        }
        let Some(rest) = line.strip_prefix("data:") else {
            // `event:`, `id:`, `retry:` fields are not used by the chat API.
            return;
        };
        let payload = rest.strip_prefix(' ').unwrap_or(rest);
        if self.has_data {
            self.data.push('\n');
        }
        self.data.push_str(payload);
        self.has_data = true;
    }
}

#[cfg(test)]
mod tests {
    use super::{SseParser, SseTooLarge, MAX_EVENT_BYTES};

    #[test]
    fn parses_events_across_chunk_boundaries() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: {\"a\":1}").unwrap().is_empty());
        assert_eq!(
            parser.push(b"\r\n\r\nda").unwrap(),
            vec!["{\"a\":1}".to_string()]
        );
        let mut parser = SseParser::new();
        let events = parser.push(b"data: hello\n\ndata: world\n\n").unwrap();
        assert_eq!(events, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let mut parser = SseParser::new();
        let events = parser
            .push(b": keep-alive\nevent: message\nid: 1\ndata: x\n\n")
            .unwrap();
        assert_eq!(events, vec!["x".to_string()]);
    }

    #[test]
    fn joins_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: a\ndata: b\n\n").unwrap();
        assert_eq!(events, vec!["a\nb".to_string()]);
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: tail").unwrap().is_empty());
        assert_eq!(parser.finish().unwrap(), vec!["tail".to_string()]);
    }

    #[test]
    fn rejects_oversized_pending_lines() {
        let mut parser = SseParser::new();
        let oversized = vec![b'a'; MAX_EVENT_BYTES + 1];
        assert_eq!(parser.push(&oversized), Err(SseTooLarge));
    }

    #[test]
    fn rejects_oversized_event_data() {
        let mut parser = SseParser::new();
        let line = format!("data: {}\n", "a".repeat(MAX_EVENT_BYTES + 1));
        assert_eq!(parser.push(line.as_bytes()), Err(SseTooLarge));
    }
}
