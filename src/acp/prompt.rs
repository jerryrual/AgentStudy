//! Conversion of stable ACP prompt content into the current text-only model.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "consumed by the upcoming ACP prompt handler")
)]

use agent_client_protocol::v1::schema::ContentBlock;
use anyhow::{bail, ensure, Result};

use crate::model::Message;

/// Convert baseline ACP prompt blocks into one internal user message.
pub(crate) fn user_message(blocks: &[ContentBlock]) -> Result<Message> {
    let mut text = Vec::with_capacity(blocks.len());
    let mut has_meaningful_content = false;
    for block in blocks {
        match block {
            ContentBlock::Text(content) => {
                has_meaningful_content |= !content.text.trim().is_empty();
                text.push(content.text.clone());
            }
            ContentBlock::ResourceLink(link) => {
                let label = link.title.as_deref().unwrap_or(&link.name);
                has_meaningful_content |= !label.trim().is_empty() || !link.uri.trim().is_empty();
                text.push(format!("[resource: {label} <{}>]", link.uri));
            }
            ContentBlock::Image(_) => bail!("unsupported ACP prompt content block: image"),
            ContentBlock::Audio(_) => bail!("unsupported ACP prompt content block: audio"),
            ContentBlock::Resource(_) => bail!("unsupported ACP prompt content block: resource"),
            _ => bail!("unsupported ACP prompt content block: unknown"),
        }
    }
    ensure!(
        has_meaningful_content,
        "ACP prompt must contain non-empty content"
    );
    Ok(Message::user(text.join("\n\n")))
}

#[cfg(test)]
mod tests {
    use agent_client_protocol::v1::schema::{
        Annotations, AudioContent, BlobResourceContents, ContentBlock, EmbeddedResource,
        EmbeddedResourceResource, ImageContent, Meta, ResourceLink, Role, TextContent,
        TextResourceContents,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::session::SessionStore;

    #[test]
    fn projects_baseline_blocks_in_order_without_trimming_text() {
        let blocks = vec![
            ContentBlock::from(" first "),
            ContentBlock::ResourceLink(
                ResourceLink::new("notes.md", "file:///notes.md").title("Notes"),
            ),
            ContentBlock::from("third\nline"),
            ContentBlock::ResourceLink(ResourceLink::new("fallback.md", "file:///fallback.md")),
        ];

        assert_eq!(
            user_message(&blocks).unwrap(),
            Message::user(
                " first \n\n[resource: Notes <file:///notes.md>]\n\nthird\nline\n\n[resource: fallback.md <file:///fallback.md>]"
            )
        );
    }

    #[test]
    fn annotations_and_metadata_do_not_change_text_payloads() {
        let annotations = Annotations::new()
            .audience(vec![Role::Assistant, Role::User])
            .last_modified("2026-11-07T00:00:00Z")
            .priority(0.5)
            .meta(Meta::from_iter([(
                "hint".to_string(),
                serde_json::json!(true),
            )]));
        let block =
            ContentBlock::Text(TextContent::new("unchanged").annotations(annotations).meta(
                Meta::from_iter([("source".to_string(), serde_json::json!("client"))]),
            ));

        assert_eq!(user_message(&[block]).unwrap(), Message::user("unchanged"));
    }

    #[test]
    fn rejects_prompts_without_meaningful_baseline_content() {
        for blocks in [
            Vec::new(),
            vec![ContentBlock::from("")],
            vec![ContentBlock::from(" \n"), ContentBlock::from("\t")],
            vec![ContentBlock::ResourceLink(
                ResourceLink::new("name is ignored", " \n").title("\t"),
            )],
        ] {
            let error = user_message(&blocks).unwrap_err();
            assert_eq!(
                error.to_string(),
                "ACP prompt must contain non-empty content"
            );
        }
    }

    #[test]
    fn retains_an_empty_link_projection_when_other_content_is_meaningful() {
        let blocks = [
            ContentBlock::from("kept"),
            ContentBlock::ResourceLink(ResourceLink::new(" ", "\t")),
        ];

        assert_eq!(
            user_message(&blocks).unwrap(),
            Message::user("kept\n\n[resource:   <\t>]")
        );
    }

    #[test]
    fn rejects_every_unsupported_content_kind_without_partial_conversion() {
        let unsupported = [
            (
                "image",
                ContentBlock::Image(ImageContent::new("aGVsbG8=", "image/png")),
            ),
            (
                "audio",
                ContentBlock::Audio(AudioContent::new("aGVsbG8=", "audio/wav")),
            ),
            (
                "resource",
                ContentBlock::Resource(EmbeddedResource::new(
                    EmbeddedResourceResource::TextResourceContents(TextResourceContents::new(
                        "context",
                        "file:///context.txt",
                    )),
                )),
            ),
            (
                "resource",
                ContentBlock::Resource(EmbeddedResource::new(
                    EmbeddedResourceResource::BlobResourceContents(BlobResourceContents::new(
                        "aGVsbG8=",
                        "file:///context.bin",
                    )),
                )),
            ),
        ];

        for (kind, block) in unsupported {
            let error = user_message(&[
                ContentBlock::from("before"),
                block,
                ContentBlock::from("after"),
            ])
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("unsupported ACP prompt content block: {kind}")
            );
        }
    }

    #[test]
    fn one_acp_prompt_persists_as_one_user_message() {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let session = store.create("/workspace", "model-a", "build").unwrap();
        let message =
            user_message(&[ContentBlock::from("first"), ContentBlock::from("second")]).unwrap();

        session.append_message(&message).unwrap();
        let loaded = store.load(session.id()).unwrap();

        assert_eq!(loaded.messages(), vec![Message::user("first\n\nsecond")]);
    }
}
