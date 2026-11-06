mod common;

use anyhow::Result;
use common::*;
use serde_json::{json, Value};

use icode::compaction_archive::{ArchiveRecord, ArchiveStore};
use icode::tools::{ToolContext, ToolRegistry};

async fn run(
    registry: &ToolRegistry,
    name: &str,
    arguments: Value,
    context: &ToolContext,
) -> Result<String> {
    let tool = registry.get(name).expect("registered tool");
    tool.run(&arguments, context).await
}

fn registry() -> ToolRegistry {
    ToolRegistry::builtin()
}

#[tokio::test]
async fn read_file_numbers_lines_and_pages() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("file.txt"), "one\ntwo\nthree\n").unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "read_file",
        json!({"path": "file.txt"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("1\tone"), "{output}");
    assert!(output.contains("3\tthree"), "{output}");

    let output = run(
        &registry(),
        "read_file",
        json!({"path": "file.txt", "offset": 2, "limit": 1}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("2\ttwo"), "{output}");
    assert!(!output.contains("1\tone"), "{output}");
    assert!(output.contains("more lines"), "{output}");
}

#[tokio::test]
async fn read_file_retrieves_an_archived_record_through_its_catalog() {
    let directory = tempfile::tempdir().unwrap();
    let workspace = directory.path().join("workspace");
    let archive_root = directory.path().join("session archive");
    std::fs::create_dir(&workspace).unwrap();
    let store = ArchiveStore::with_root(&archive_root);
    store
        .archive_round(&[ArchiveRecord {
            turn: 1,
            round: 2,
            sequence: 3,
            tool: "read_file".into(),
            created_at: "2026-11-05T14:20:00Z".into(),
            kind: "group".into(),
            group_id: "call_archived".into(),
            session_id: "session-id".into(),
            markdown: "## Tool result\n\narchived retrieval detail\n".into(),
        }])
        .unwrap();
    let catalog_pointer = store.catalog_pointer_text().unwrap().unwrap();
    let catalog_path = catalog_pointer
        .split("; catalog: ")
        .nth(1)
        .unwrap()
        .strip_suffix(" (contains each record's relative path).")
        .unwrap();
    let context = tool_context(&workspace);
    let registry = registry();

    let catalog = run(
        &registry,
        "read_file",
        json!({"path": catalog_path}),
        &context,
    )
    .await
    .unwrap();
    let catalog_line = catalog
        .lines()
        .find_map(|line| line.split_once('\t').map(|(_, value)| value))
        .unwrap();
    let entry: Value = serde_json::from_str(catalog_line).unwrap();
    let relative_path = entry["relative_path"].as_str().unwrap();
    let pointer_root = std::path::Path::new(catalog_path)
        .parent()
        .and_then(std::path::Path::parent)
        .and_then(std::path::Path::parent)
        .expect("catalog pointer uses the session archive layout");
    let record_path = pointer_root.join(relative_path);
    let record = run(
        &registry,
        "read_file",
        json!({"path": record_path}),
        &context,
    )
    .await
    .unwrap();

    assert!(
        catalog.contains("compactions/dropped/turn001/"),
        "{catalog}"
    );
    assert!(record.contains("archived retrieval detail"), "{record}");
    assert!(record.contains("<!-- end of record -->"), "{record}");
}

#[tokio::test]
async fn read_file_rejects_binary_files() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("bin.dat"), [0u8, 1, 2, 3]).unwrap();
    let context = tool_context(directory.path());
    let error = run(
        &registry(),
        "read_file",
        json!({"path": "bin.dat"}),
        &context,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("binary"), "{error}");
}

#[tokio::test]
async fn write_file_creates_parent_directories() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "write_file",
        json!({"path": "nested/dir/hello.txt", "content": "hi there\n"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("Wrote nested/dir/hello.txt"), "{output}");
    let written = std::fs::read_to_string(directory.path().join("nested/dir/hello.txt")).unwrap();
    assert_eq!(written, "hi there\n");
}

#[tokio::test]
async fn edit_file_replaces_unique_text_and_reports_ambiguity() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("code.rs"), "let x = 1;\nlet y = 2;\n").unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "edit_file",
        json!({"path": "code.rs", "old_string": "let x = 1;", "new_string": "let x = 42;"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("Edited code.rs"), "{output}");
    assert_eq!(
        std::fs::read_to_string(directory.path().join("code.rs")).unwrap(),
        "let x = 42;\nlet y = 2;\n"
    );

    let error = run(
        &registry(),
        "edit_file",
        json!({"path": "code.rs", "old_string": "let", "new_string": "const"}),
        &context,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("occurs 2 times"), "{error}");

    let output = run(
        &registry(),
        "edit_file",
        json!({"path": "code.rs", "old_string": "let", "new_string": "const", "replace_all": true}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("2 replacement"), "{output}");
    let replaced = std::fs::read_to_string(directory.path().join("code.rs")).unwrap();
    assert!(!replaced.contains("let "), "{replaced}");
    assert!(replaced.contains("const x = 42;"), "{replaced}");
}

#[tokio::test]
async fn list_dir_walks_the_tree() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src/bin")).unwrap();
    std::fs::write(directory.path().join("src/main.rs"), "fn main() {}").unwrap();
    std::fs::write(directory.path().join("src/bin/tool.rs"), "fn main() {}").unwrap();
    std::fs::create_dir_all(directory.path().join(".git")).unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "list_dir",
        json!({"path": ".", "depth": 2}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("src/"), "{output}");
    assert!(output.contains("main.rs"), "{output}");
    assert!(output.contains("bin/"), "{output}");
    assert!(!output.contains(".git"), "{output}");
    // The tree must keep parents before their indented children.
    let parent = output.find("src/").expect("src/ in listing");
    let child = output.find("  bin/").expect("indented bin/ in listing");
    let file = output
        .find("  main.rs")
        .expect("indented main.rs in listing");
    assert!(parent < child, "parent before children: {output}");
    assert!(parent < file, "parent before files: {output}");
}

#[tokio::test]
async fn grep_finds_matches_with_paths() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("src/main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "grep",
        json!({"pattern": "println", "path": "."}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("src/main.rs:2:"), "{output}");
    assert!(output.contains("println"), "{output}");

    let output = run(
        &registry(),
        "grep",
        json!({"pattern": "missing-pattern", "path": "."}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("No matches"), "{output}");
}

#[tokio::test]
async fn glob_finds_files() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("src/main.rs"), "").unwrap();
    std::fs::write(directory.path().join("src/lib.rs"), "").unwrap();
    std::fs::write(directory.path().join("README.md"), "").unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "glob",
        json!({"pattern": "**/*.rs", "path": "."}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("src/main.rs"), "{output}");
    assert!(output.contains("src/lib.rs"), "{output}");
    assert!(!output.contains("README.md"), "{output}");
}

#[tokio::test]
async fn bash_reports_exit_codes_and_stderr() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "bash",
        json!({"command": "printf 'hello'; printf 'oops' >&2"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("exit code: 0"), "{output}");
    assert!(output.contains("hello"), "{output}");
    assert!(output.contains("stderr:\noops"), "{output}");

    let output = run(&registry(), "bash", json!({"command": "exit 3"}), &context)
        .await
        .unwrap();
    assert!(output.contains("exit code: 3"), "{output}");
}

#[tokio::test]
async fn bash_honours_its_timeout() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let error = run(
        &registry(),
        "bash",
        json!({"command": "echo before-timeout; sleep 5", "timeout_secs": 1}),
        &context,
    )
    .await
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("timed out"), "{message}");
    assert!(
        message.contains("before-timeout"),
        "partial output should be kept: {message}"
    );
}

#[tokio::test]
async fn bash_caps_captured_output() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "bash",
        json!({"command": "yes a | head -c 100000"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("truncated at"), "{output}");
    assert!(output.len() < 70_000, "output should be capped");
}

#[tokio::test]
async fn todo_write_updates_the_session_list() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "todo_write",
        json!({"todos": [
            {"content": "read the code", "status": "completed"},
            {"content": "write the test", "status": "in_progress"},
            {"content": "run the tests"}
        ]}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("[x] read the code"), "{output}");
    assert!(output.contains("[~] write the test"), "{output}");
    assert!(output.contains("[ ] run the tests"), "{output}");
    assert_eq!(context.todos.lock().await.len(), 3);
}

#[tokio::test]
async fn ask_user_returns_the_scripted_answer() {
    let directory = tempfile::tempdir().unwrap();
    let context = tool_context(directory.path());
    let output = run(
        &registry(),
        "ask_user",
        json!({"question": "which database?"}),
        &context,
    )
    .await
    .unwrap();
    assert!(output.contains("scripted answer"), "{output}");
}
