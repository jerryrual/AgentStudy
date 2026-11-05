# 与 iCode 的能力对照

本文件说明 Rust 版本目前实现了 [openJiuwen-ai/iCode](https://github.com/openJiuwen-ai/iCode)
的哪些能力，以及哪些能力尚未移植。目标是明确边界，而不是承诺完整对齐。

## 已移植（v0.1）

| iCode 能力 | Rust 版实现 | 所在模块 |
| --- | --- | --- |
| 终端 TUI 对话 | ratatui 实现：对话区、输入框、状态栏、审批弹窗、帮助面板 | `src/tui/` |
| 流式回答 | SSE 增量解析，逐 token 渲染 | `src/model/sse.rs`、`client.rs` |
| Markdown 渲染 | 标题、列表、行内代码、粗体、链接、代码块 | `src/markdown.rs` |
| 模型配置 | 多份 OpenAI 兼容配置，`/model` 切换，环境变量/命令取密钥；可选 `context_window` 派生 token 预算并按估算剩余上下文下调请求输出上限 | `src/config.rs`、`src/model/` |
| 上下文裁剪（Phase 1 / 2 / 3） | 请求输入跨过派生触发阈值后，在仅供该次请求使用的投影中，Phase 1 按从旧到新的顺序将已完成旧轮次的工具结果替换为确定性摘要；仅当仍超过目标预算时，Phase 2 才按从旧到新的顺序移除已完成旧轮次的工具调用/结果批次；若此后仍超过目标预算，Phase 3 才按从旧到新的顺序将已完成的历史轮次替换为确定性的压缩上下文摘要。当前轮次、JSONL 中的权威历史及 `/resume` 恢复历史、TUI 历史和手动 `/compact` 历史保持原样。这是 Rust 版请求投影语义，不表示与上游的持久化压缩行为完全一致 | `src/context_trim.rs`、`src/model/client.rs` |
| 工具调用 | 多轮工具循环；纯只读工具批次并行执行，结果按调用顺序回填；超长输出截断 | `src/agent/` |
| 文件读写 | `read_file`、`write_file`、`edit_file`（精确匹配替换） | `src/tools/fs.rs` |
| 代码搜索 | `grep`（正则）、`glob`、`list_dir`，遵循 .gitignore | `src/tools/search.rs` |
| Shell | `bash` 工具，超时与 `kill_on_drop` | `src/tools/shell.rs` |
| 审批 | ask / auto / deny，写操作与 Shell 需确认 | `src/interaction.rs` |
| 计划清单 | `todo_write` | `src/tools/misc.rs` |
| 提问 | `ask_user`（TUI 中等待回答） | `src/tools/misc.rs` |
| 会话 | JSONL 追加写入、列举、继续；`/compact` 生成并持久化续写摘要 | `src/session.rs`、`src/compaction.rs` |
| 项目记忆 | 主 agent 从当前工作目录加载 `AGENTS.md` 并注入系统上下文 | `src/memory.rs` |
| 无界面运行 | `icode run`，JSON 输出 | `src/run.rs` |
| 直接 Shell | 输入 `!命令` 直接执行 | `src/tui/app.rs` |
| 深度推理 | 展示 `reasoning_content` 摘要 | `src/tui/ui.rs` |
| 模型协议 | Chat Completions；核心无状态 Responses API（文本、推理摘要、本地函数工具、流式/非流式） | `src/model/` |
| 重试 | 429/5xx/transport 指数退避重试 | `src/model/client.rs` |

## 尚未移植

以下按依赖顺序推进；各项列出尚未实现的部分。

### 1. 上下文 / 协议

1. 上游风格的自动持久化摘要压缩（包括轮次结束后的强制行为）和 Last Words；当前已实现 Phase 1 / 2 / 3 的 Rust 请求投影，但这不表示与上游的持久化压缩行为完全一致。持久化摘要压缩仍仅支持手动 `/compact`，其摘要输入超限时会优先保留最近内容。
2. Responses API 的服务端续传、后台响应与托管工具。
3. 图片输入、图片压缩与多模态消息持久化。

### 2. 工具 / 扩展

1. MCP 服务器与 Skills。
2. Hooks 与子智能体；依赖统一的工具协议和上下文隔离。
3. `web_search` / `web_fetch` 等联网工具；依赖托管工具或独立搜索提供方。
4. 真正的交互式 Shell 终端；当前 `!` 命令只回显输出，不接管全屏程序。

### 3. 可观测性 / 恢复

1. 内建 diff 查看器与轮次回滚。
2. 轨迹（trajectory）及 token / 耗时分析面板。

### 4. 集成 / 工作流

1. ACP 服务（`icode acp` / `icode serve`）与编辑器集成。
2. 工作流（workflow）编排与节点可视化；依赖扩展协议和轨迹能力。

### 5. UX / 分发

1. 模型配置的 TUI 编辑（F2 / F4）。
2. 多语言界面与主题切换。
3. Windows / Linux 安装包与自更新。

## 有意保留差异

- 配置格式为单个 `settings.json`，不是 iCode 的多文件模型配置；字段命名尽量贴近，
  便于迁移。
- 会话目录默认在系统数据目录下（`ICODE_HOME` 可覆盖），iCode 的会话格式不同，
  不能直接互相读取。
- 模型重试策略固定为 3 次指数退避；可通过 `ICODE_RETRY_BASE_MS` 调整基准间隔（测试用）。
- 工具并发仅对 `read_file`、`list_dir`、`grep`、`glob` 开放；写操作、Shell、提问与
  todo 更新保持顺序执行，以保证审批、交互和会话状态的确定性。
- Responses API 采用无状态完整历史重放；不支持已存储响应续传、服务端托管工具（如
  `web_search`）或图片输入，遇到未支持的可执行输出类型会失败关闭。
- `icode run` 与 iCode 一样绕过审批；`--approval deny` 提供只读保护。

## 设计取向

1. **核心可读**：每个模块只负责一件事，事件流（`EventSink`）把智能体与界面解耦，
   便于替换渲染层或接入其他前端。
2. **可测试**：HTTP 层通过本地 SSE mock 服务测试，TUI 通过 `TestBackend` 渲染测试，
   会话与工具都有单元/集成测试。
3. **安全默认**：写操作默认逐次审批；密钥不落日志；`icode run` 的写操作边界显式可控。
