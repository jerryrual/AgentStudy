//! Incremental Server-Sent-Events parser for the streaming API.
//!
//! The parser is deliberately independent from HTTP so it can be unit tested
//! with arbitrary chunk boundaries.

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
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
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
        events
    }

    /// Flush the trailing event at end of stream, if the server omitted the
    /// final blank line.
    pub fn finish(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        if !self.buffer.is_empty() {
            let line = String::from_utf8_lossy(&self.buffer).to_string();
            self.buffer.clear();
            self.consume_line(&line, &mut events);
        }
        if self.has_data {
            events.push(std::mem::take(&mut self.data));
            self.has_data = false;
        }
        events
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
    use super::SseParser;

    #[test]
    fn parses_events_across_chunk_boundaries() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: {\"a\":1}").is_empty());
        assert_eq!(parser.push(b"\r\n\r\nda"), vec!["{\"a\":1}".to_string()]);
        let mut parser = SseParser::new();
        let events = parser.push(b"data: hello\n\ndata: world\n\n");
        assert_eq!(events, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let mut parser = SseParser::new();
        let events = parser.push(b": keep-alive\nevent: message\nid: 1\ndata: x\n\n");
        assert_eq!(events, vec!["x".to_string()]);
    }

    #[test]
    fn joins_multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.push(b"data: a\ndata: b\n\n");
        assert_eq!(events, vec!["a\nb".to_string()]);
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut parser = SseParser::new();
        assert!(parser.push(b"data: tail").is_empty());
        assert_eq!(parser.finish(), vec!["tail".to_string()]);
    }
}
