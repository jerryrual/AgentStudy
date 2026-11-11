# icode（Rust 版）

用 Rust 重新实现 [openJiuwen-ai/iCode](https://github.com/openJiuwen-ai/iCode) 的核心能力：
一个运行在终端里的编码智能体（coding agent）——TUI 对话、模型接入、流式输出、
工具调用与审批、会话持久化，以及无界面的一次性任务执行。

> 本仓库当前是 **核心功能的 Rust 版本（v0.1）**，并非 iCode 的完整移植。
> 已移植与尚未移植的能力见 [docs/porting.md](docs/porting.md)。

## 特性

- **TUI 对话界面**：基于 ratatui，包含对话视图、输入框、状态栏（模型 / 审批模式 /
  工作目录 / token / 会话），支持流式 Markdown 渲染、工具调用卡片与审批弹窗。
- **模型接入**：兼容 OpenAI Chat Completions 或核心 Responses API 的服务，
  支持流式 SSE、推理摘要、token 用量统计、429/5xx 自动重试和本地函数工具；Responses
  模式支持流式/非流式文本、显式 opt-in 的 provider-hosted `web_search` / `file_search`、默认关闭的
  provider 存储续传，以及 OpenAI 非流式后台执行，但不支持后台流式恢复、远端取消、其他托管工具或图片。
- **智能体循环**：系统提示词 + 多轮工具调用；只读检索批次并行执行并按调用顺序回填，
  写操作与交互工具保持顺序执行；内置 Code 与只读 Q&A 两种智能体。
- **工具集**：`read_file`、`write_file`、`edit_file`、`list_dir`、`grep`、`glob`、
  `bash`、`todo_write`、`ask_user`。
- **审批策略**：`ask`（默认，逐次确认写操作与 Shell）、`auto`、`deny`。
- **会话与记忆**：JSONL 追加写入，支持 `--continue`、`/sessions`、`/resume <id>`；
  `/compact` 生成并持久化续写摘要；主 agent 自动将当前工作目录的 `AGENTS.md` 作为
  项目参考上下文加载，文件缺失时静默跳过。
- **无界面运行**：`icode run "..."`，支持 `--json` 供脚本消费。
- **输入便捷**：`/` 命令、`!` 直接执行 Shell、`@file` 附加文件内容。

## 界面

真实终端截图（118×40，DeepSeek 兼容的 mock 模型流式输出）：

![TUI 对话：流式回答、Markdown 渲染与工具调用卡片](docs/screenshots/tui-answer.png)

![执行写操作前的审批弹窗](docs/screenshots/tui-approval.png)

![批准后工具执行结果与最终回答](docs/screenshots/tui-tool-result.png)

![手动压缩后以持久化摘要继续会话](docs/screenshots/compaction-after.png)

## 构建

需要 Rust 1.88 或更新版本。

```bash
cargo build --release
./target/release/icode --help
```

## 配置

首次运行会自动创建 `settings.json`（Linux：`~/.config/icode/settings.json`，
macOS：`~/Library/Application Support/icode/settings.json`），也可以用
`ICODE_HOME` 指定配置与会话目录：

```bash
export ICODE_HOME="$HOME/.icode"
```

默认配置（DeepSeek）：

```json
{
  "models": [
    {
      "name": "deepseek",
      "provider": "deepseek",
      "api_style": "chat",
      "model": "deepseek-chat",
      "base_url": "https://api.deepseek.com",
      "api_key": "{{DEEPSEEK_API_KEY}}",
      "context_window": 128000,
      "max_tokens": 8192,
      "temperature": 0.1,
      "stream": true
    }
  ],
  "current_model": "deepseek",
  "approval_mode": "ask",
  "agent": "code",
  "max_tool_rounds": 40
}
```

模型配置可选填 `context_window`，表示该模型的总上下文窗口（token）。与 `max_tokens` 同时
设置后，每次请求都会以 UTF-8 字节数保守估算系统提示、对话历史和工具定义占用的 token，
并将 `max_tokens`（Responses API 中为 `max_output_tokens`）下调到估算的剩余上下文以内；
该限制不会调高配置的 `max_tokens`，Chat 与 Responses 对同一语义上下文使用相同估算。
Responses profile 可设置 `"responses_web_search": true`，让 provider 在普通请求中运行托管
`web_search`；该搜索不经过本地工具审批或网络策略，其 native call 会随会话持久化并在无状态请求中回放。
也可配置 `"responses_file_search": {"vector_store_ids": ["vs_..."], "max_num_results": 20}`，让
provider 搜索既有 vector store；`max_num_results` 可省略，若设置须为 1–50。文件搜索结果、原生
`file_search_call` 与文件引用注解会进入本地会话并原样回放，但当前 TUI 仅显示回答文本，不单独渲染引用。
这些托管调用都不经过本地工具审批/执行；切换配置会使存储续传建立新 scope。Phase 4 Last Words
side-call 始终不暴露托管工具。还可显式设置
`"responses_stored_continuation": true`：普通 Responses 结果会由 provider 远端存储，Rust 会把不透明
response id 与 endpoint、model、认证身份、当前 system prompt 的 SHA-256 scope 及本地输入前缀摘要一起绑定到
assistant 会话记录；scope 与前缀均未变化且没有本地裁剪/投影时，下一次请求只发送新增后缀并携带
`previous_response_id`，否则完整回放并建立新锚点。该选项默认关闭；启用后数据受 provider 的远端保留策略约束，删除本地会话不会删除远端 response，
过期或被 provider 拒绝的 response id 会显式报错。OpenAI profile 还可在上述存储续传已启用且
`"stream": false` 时设置 `"responses_background": true`：每个普通请求只创建一次后台 response，随后每
2 秒轮询同一 id，最多等待 10 分钟；轮询 GET 的临时错误可重试，但 create 的 transport/429/5xx 不会
自动重发，以免重复执行或计费。超时或本地取消会停止等待但不会远端取消，provider 任务可能继续运行并
按其保留策略存储；Zero Data Retention 配置通常不兼容该模式。Last Words 始终 `store: false` 且不进入
后台或远端续传链。
当请求输入估算达到或超过由模型配置派生的触发阈值时，当前 Rust 实现会在仅供该次请求使用的投影中执行
分阶段裁剪：Phase 1 按从旧到新的顺序，将已完成旧轮次的工具结果替换为确定性摘要；仅当输入
仍超过目标预算时，Phase 2 才按从旧到新的顺序移除已完成旧轮次的工具调用/结果批次；若此后仍
超过目标预算，Phase 3 才按从旧到新的顺序，将已完成的历史轮次替换为确定性的压缩上下文摘要。
Phase 1 / 2 仍仅改变当次请求；Phase 3 的折叠批次会在 provider 调用前原子追加到 JSONL，写入
失败则不会发起该次 provider 调用。TUI 显示历史保持原样，取消已开始的轮次后也会从已持久化的
权威历史恢复模型上下文。成功轮次结束后，如果最终一次模型调用报告的输入 token 严格超过派生的 force 阈值，
TUI 和 `icode run` 会自动将一个最旧的先前已完成轮次持久化为确定性摘要；刚完成轮次和所有较新
历史保持不变，原始 JSONL 记录也不会删除。持久化检查点会先写入成功，再替换内存中的模型上下文；
写入失败时保留完整上下文。若 provider 以严格的上下文溢出错误拒绝 agent 请求，客户端会绕过常规触发阈值，
强制运行现有 Phase 1 / 2 / 3；仅当保守估算的输入上下文严格缩小时才立即重试一次。第二次溢出直接
返回，且 provider 明示的窗口小于配置值时不会盲目重发。若 Phase 1–3 后仍超过目标且当前轮次
包含工具调用，客户端会发起一次无工具、非流式、输出受限的 Last Words side-call；仅接受完整的
`Task` / `Progress` / `Next` note。note 必须让正常请求的保守输入估算严格下降，并与当前轮次替换
作为一条 `ContextProjection` 原子持久化，之后才会改变内存历史并继续同一 agent round。provider
明示溢出时也可强制走此路径；若主动 Last Words 未提交投影、随后原始请求溢出，会在同一 turn breaker
下再尝试一次 Last Words；成功投影后的 provider 请求仍不递归恢复，第二次外层溢出直接返回。生成、校验、
持久化或取消失败都会保留原始工具历史，TUI 通过事件队列屏障保证先前 assistant/tool 记录先落盘。Chat Completions 和无状态
Responses API 都使用同一语义。模型配置可选填 `last_words_supplement`（例如
`"优先保留复现步骤、已排除原因和待验证假设。"`），在生成 Last Words 时强调容易丢失但对续写重要的信息；
它只进入 Last Words side-call，不进入普通模型请求，最多 4000 个字符。模型配置还可用
`last_words_max_output_tokens` 调整 side-call 输出上限（1000–64000，默认 1024）；实际值不会超过模型的
`max_tokens`，也会继续受剩余上下文预算限制。

API 密钥有三种写法：

| 写法 | 含义 |
| --- | --- |
| `"sk-..."` | 直接写在配置里 |
| `"{{DEEPSEEK_API_KEY}}"` | 读取环境变量 |
| `"{{!token-provider get}}"` | 运行命令，取 stdout（30 秒超时） |

配置好后运行：

```bash
export DEEPSEEK_API_KEY=sk-...
icode                      # 打开 TUI
icode run "解释这个项目的结构" --agent qa
icode run "修复测试并运行 cargo test" --json
icode models               # 查看已配置的模型
```

## 使用

### TUI

| 按键 | 作用 |
| --- | --- |
| `Enter` | 发送消息 / 回答提问 |
| `Esc` | 取消正在运行的一轮 / 关闭弹窗 |
| `Ctrl-C` | 运行中取消，空闲时退出 |
| `Up` / `Down` | 历史输入 |
| `PgUp` / `PgDn` | 滚动对话 |
| `Tab` | 补全 `/` 命令 |

常用命令：`/help`、`/new`、`/models`、`/model <name>`、`/approval ask|auto|deny`、
`/sessions`、`/resume <id>`、`/compact`、`/cwd <path>`、`!<shell 命令>`、`@<文件>`。

主 agent 创建时会读取当前工作目录根部的普通文件 `AGENTS.md`，并按原文件整体加入系统上下文；
不搜索父目录，不加载 `ICODE.md`，也不跟随符号链接。内容会随模型请求发送给服务商，
请勿在其中放置密钥或其他敏感信息。
`AGENTS.md` 也能影响模型发起的工具调用；在不可信仓库中应先检查该文件，或以
`--approval deny` 运行无界面任务。超过约 35,000 token 预算的文件会整体跳过并显示提示。

`/compact` 会把当前完整对话作为数据交给当前模型生成续写摘要；摘要写入追加式会话文件后，
模型上下文才替换为该摘要。原始记录仍保留在 JSONL 文件中，压缩失败或被取消时不会改变当前上下文。
手动 `/compact` 不受自动阈值限制；其摘要输入超出窗口时会优先保留最近内容，使已超限会话可恢复。

### 无界面运行

```bash
icode run "总结 src/ 下各模块的职责" --cwd . --json
```

- 最终回复写入 stdout（流式），进度写入 stderr。
- `--json` 输出单个 JSON 对象：`ok`、`text`、`usage`、`session_id`、`tool_calls`、
  `rounds`、`duration_ms`。
- 与 iCode 的 `icode run` 一致，默认不弹审批；需要时传 `--approval deny` 禁止一切写操作。
- `--session <id>` 在已有会话上继续。

## 架构

```
src/
├── cli.rs        # clap 命令行入口
├── compaction.rs # 模型辅助的上下文摘要
├── config.rs     # 模型配置、设置、API Key 解析
├── model/        # OpenAI Chat Completions 客户端 + SSE 解析
├── agent/        # 工具调用循环、事件流、系统提示词
├── tools/        # 文件、搜索、Shell、待办、提问工具
├── interaction.rs# 审批与提问接口
├── session.rs    # JSONL 会话存储
├── markdown.rs   # 终端 Markdown 渲染
├── tui/          # ratatui 界面（状态、输入、渲染）
└── run.rs        # icode run 无界面模式
```

## 测试

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

集成测试自带一个本地 SSE mock 模型服务，覆盖流式输出、工具调用、审批、拒绝、
重试、轮次上限，以及真实 CLI 的端到端 `icode run --json`。

## 许可证

Apache-2.0，与 iCode 相同。
