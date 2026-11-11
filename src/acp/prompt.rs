//! Conversion and validation of stable ACP prompt content.

use agent_client_protocol::v1::schema::ContentBlock;
use anyhow::{bail, ensure, Result};

use crate::model::types::{MAX_PROMPT_IMAGES, MAX_PROMPT_IMAGE_BYTES};
use crate::model::{ImageInput, Message, UserContentPart};

/// Convert stable ACP prompt blocks into one durable user message.
pub(crate) fn user_message(blocks: &[ContentBlock]) -> Result<Message> {
    let mut legacy_text = Vec::with_capacity(blocks.len());
    let mut parts = Vec::with_capacity(blocks.len());
    let mut has_meaningful_content = false;
    let mut has_image = false;
    let mut image_bytes = 0usize;
    let mut image_count = 0usize;
    for block in blocks {
        match block {
            ContentBlock::Text(content) => {
                has_meaningful_content |= !content.text.trim().is_empty();
                legacy_text.push(content.text.clone());
                push_text(&mut parts, content.text.clone());
            }
            ContentBlock::ResourceLink(link) => {
                let label = link.title.as_deref().unwrap_or(&link.name);
                has_meaningful_content |= !label.trim().is_empty() || !link.uri.trim().is_empty();
                let text = format!("[resource: {label} <{}>]", link.uri);
                legacy_text.push(text.clone());
                push_text(&mut parts, text);
            }
            ContentBlock::Image(image) => {
                image_count = image_count.saturating_add(1);
                ensure!(
                    image_count <= MAX_PROMPT_IMAGES,
                    "ACP prompt exceeds the {MAX_PROMPT_IMAGES}-image limit"
                );
                let remaining_bytes = MAX_PROMPT_IMAGE_BYTES.saturating_sub(image_bytes);
                let (image, decoded_bytes) = ImageInput::validated(
                    image.mime_type.clone(),
                    image.data.clone(),
                    remaining_bytes,
                )?;
                image_bytes += decoded_bytes;
                parts.push(UserContentPart::Image { image });
                has_meaningful_content = true;
                has_image = true;
            }
            ContentBlock::Audio(_) => bail!("unsupported ACP prompt content block: audio"),
            ContentBlock::Resource(_) => bail!("unsupported ACP prompt content block: resource"),
            _ => bail!("unsupported ACP prompt content block: unknown"),
        }
    }
    ensure!(
        has_meaningful_content,
        "ACP prompt must contain non-empty content"
    );
    if has_image {
        Ok(Message::user_parts(parts))
    } else {
        Ok(Message::user(legacy_text.join("\n\n")))
    }
}

fn push_text(parts: &mut Vec<UserContentPart>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(UserContentPart::Text { text: previous }) = parts.last_mut() {
        previous.push_str("\n\n");
        previous.push_str(&text);
    } else {
        parts.push(UserContentPart::Text { text });
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use agent_client_protocol::v1::schema::{
        Annotations, AudioContent, BlobResourceContents, ContentBlock, EmbeddedResource,
        EmbeddedResourceResource, ImageContent, Meta, ResourceLink, Role, TextContent,
        TextResourceContents,
    };
    use base64::prelude::{Engine as _, BASE64_STANDARD};
    use image::{DynamicImage, ImageFormat};
    use tempfile::TempDir;

    use super::*;
    use crate::model::types::MAX_IMAGE_BYTES;
    use crate::session::SessionStore;

    fn encoded_image(format: ImageFormat) -> String {
        BASE64_STANDARD.encode(image_bytes(format))
    }

    fn image_bytes(format: ImageFormat) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::new_rgba8(1, 1)
            .write_to(&mut bytes, format)
            .unwrap();
        bytes.into_inner()
    }

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
    fn preserves_interleaved_text_and_valid_images() {
        let png = encoded_image(ImageFormat::Png);
        let (expected_png, _) =
            ImageInput::validated("image/png".into(), png.clone(), MAX_IMAGE_BYTES).unwrap();
        let jpeg = encoded_image(ImageFormat::Jpeg);
        let blocks = [
            ContentBlock::from(""),
            ContentBlock::from(" \n"),
            ContentBlock::from("before"),
            ContentBlock::Image(ImageContent::new(png.clone(), "image/png")),
            ContentBlock::from("after"),
        ];

        assert_eq!(
            user_message(&blocks).unwrap(),
            Message::user_parts(vec![
                UserContentPart::Text {
                    text: " \n\n\nbefore".into(),
                },
                UserContentPart::Image {
                    image: expected_png,
                },
                UserContentPart::Text {
                    text: "after".into(),
                },
            ])
        );
        assert!(
            user_message(&[ContentBlock::Image(ImageContent::new(jpeg, "image/jpeg"))]).is_ok()
        );
    }

    #[test]
    fn rejects_invalid_image_payloads_atomically() {
        let png = image_bytes(ImageFormat::Png);
        user_message(&[ContentBlock::Image(ImageContent::new(
            BASE64_STANDARD.encode(&png),
            "image/png",
        ))])
        .unwrap();
        let invalid = vec![
            (String::new(), "image/png", "must not be empty"),
            ("not-base64".into(), "image/png", "not valid padded base64"),
            (
                BASE64_STANDARD.encode(&png),
                "image/bmp",
                "unsupported ACP image MIME type",
            ),
            (
                BASE64_STANDARD.encode(&png),
                "image/jpeg",
                "not a valid image/jpeg image",
            ),
            (
                BASE64_STANDARD.encode(&png[..8]),
                "image/png",
                "not a valid image/png image",
            ),
        ];

        for (data, media_type, expected) in invalid {
            let error = user_message(&[
                ContentBlock::from("before"),
                ContentBlock::Image(ImageContent::new(data, media_type)),
                ContentBlock::from("after"),
            ])
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[test]
    fn enforces_per_image_and_aggregate_byte_limits() {
        let oversized = "A".repeat(MAX_IMAGE_BYTES.div_ceil(3) * 4 + 1);
        let error = user_message(&[ContentBlock::Image(ImageContent::new(
            oversized,
            "image/png",
        ))])
        .unwrap_err();
        assert!(error.to_string().contains("decoded limit"), "{error:#}");

        let mut maximum = image_bytes(ImageFormat::Png);
        maximum.resize(MAX_IMAGE_BYTES, 0);
        let maximum = BASE64_STANDARD.encode(maximum);
        let tiny = encoded_image(ImageFormat::Png);
        let error = user_message(&[
            ContentBlock::Image(ImageContent::new(maximum.clone(), "image/png")),
            ContentBlock::Image(ImageContent::new(maximum, "image/png")),
            ContentBlock::Image(ImageContent::new(tiny, "image/png")),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("aggregate limit"), "{error:#}");

        let tiny = encoded_image(ImageFormat::Png);
        let images = (0..=MAX_PROMPT_IMAGES)
            .map(|_| ContentBlock::Image(ImageContent::new(tiny.clone(), "image/png")))
            .collect::<Vec<_>>();
        let error = user_message(&images).unwrap_err();
        assert!(error.to_string().contains("image limit"), "{error:#}");
    }

    #[test]
    fn rejects_every_unsupported_content_kind_without_partial_conversion() {
        let unsupported = [
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
    fn one_multimodal_acp_prompt_persists_as_one_user_turn() {
        let directory = TempDir::new().unwrap();
        let store = SessionStore::with_root(directory.path().join("sessions"));
        let session = store.create("/workspace", "model-a", "build").unwrap();
        let message = user_message(&[
            ContentBlock::from("before"),
            ContentBlock::Image(ImageContent::new(
                encoded_image(ImageFormat::Png),
                "image/png",
            )),
            ContentBlock::from("after"),
        ])
        .unwrap();

        session.append_message(&message).unwrap();
        let loaded = store.load(session.id()).unwrap();

        assert_eq!(loaded.messages(), vec![message]);
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
