# 与 iCode 的能力对照

本文件说明 Rust 版本目前实现了 [openJiuwen-ai/iCode](https://github.com/openJiuwen-ai/iCode)
的哪些能力，以及哪些能力尚未移植。目标是明确边界，而不是承诺完整对齐。

## 已移植（v0.1）

| iCode 能力 | Rust 版实现 | 所在模块 |
| --- | --- | --- |
| 终端 TUI 对话 | ratatui 实现：对话区、输入框、状态栏、审批弹窗、帮助面板 | `src/tui/` |
| 流式回答 | SSE 增量解析，逐 token 渲染 | `src/model/sse.rs`、`client.rs` |
| Markdown 渲染 | 标题、列表、行内代码、粗体、链接、代码块 | `src/markdown.rs` |
| 模型配置 | 多份 OpenAI 兼容配置，`/model` 切换，环境变量/命令取密钥 | `src/config.rs` |
| 工具调用 | 多轮工具循环，结果回填，超长输出截断 | `src/agent/` |
| 文件读写 | `read_file`、`write_file`、`edit_file`（精确匹配替换） | `src/tools/fs.rs` |
| 代码搜索 | `grep`（正则）、`glob`、`list_dir`，遵循 .gitignore | `src/tools/search.rs` |
| Shell | `bash` 工具，超时与 `kill_on_drop` | `src/tools/shell.rs` |
| 审批 | ask / auto / deny，写操作与 Shell 需确认 | `src/interaction.rs` |
| 计划清单 | `todo_write` | `src/tools/misc.rs` |
| 提问 | `ask_user`（TUI 中等待回答） | `src/tools/misc.rs` |
| 会话 | JSONL 追加写入、列举、继续 | `src/session.rs` |
| 无界面运行 | `icode run`，JSON 输出 | `src/run.rs` |
| 直接 Shell | 输入 `!命令` 直接执行 | `src/tui/app.rs` |
| 深度推理 | 展示 `reasoning_content` 摘要 | `src/tui/ui.rs` |
| 重试 | 429/5xx/transport 指数退避重试 | `src/model/client.rs` |

## 尚未移植

以下能力属于 iCode 的完整形态，本版本未实现（欢迎后续补充）：

- ACP 服务（`icode acp` / `icode serve`）与编辑器集成。
- 工作流（workflow）编排与节点可视化。
- MCP 服务器、Skills、Hooks、子智能体、记忆（memory）。
- 内建 diff 查看器、轮次回滚、轨迹（trajectory）与 token/耗时分析面板。
- 真正的交互式 Shell 终端（当前 `!` 命令只回显输出，不接管全屏程序）。
- Responses API（配置中可声明，运行时会明确报错）。
- 图片/视觉输入、`web_search`、模型配置的 TUI 编辑（F2/F4）。
- 上下文压缩（`/compact` 目前只提示未实现）。
- 多语言界面、主题切换、Windows/Linux 安装包与自更新。
- 并行工具调用（当前逐个顺序执行，以保证审批顺序确定）。

## 有意保留差异

- 配置格式为单个 `settings.json`，不是 iCode 的多文件模型配置；字段命名尽量贴近，
  便于迁移。
- 会话目录默认在系统数据目录下（`ICODE_HOME` 可覆盖），iCode 的会话格式不同，
  不能直接互相读取。
- 模型重试策略固定为 3 次指数退避；可通过 `ICODE_RETRY_BASE_MS` 调整基准间隔（测试用）。
- `icode run` 与 iCode 一样绕过审批；`--approval deny` 提供只读保护。

## 设计取向

1. **核心可读**：每个模块只负责一件事，事件流（`EventSink`）把智能体与界面解耦，
   便于替换渲染层或接入其他前端。
2. **可测试**：HTTP 层通过本地 SSE mock 服务测试，TUI 通过 `TestBackend` 渲染测试，
   会话与工具都有单元/集成测试。
3. **安全默认**：写操作默认逐次审批；密钥不落日志；`icode run` 的写操作边界显式可控。
