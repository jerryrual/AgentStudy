# icode（Rust 版）

用 Rust 重新实现 [openJiuwen-ai/iCode](https://github.com/openJiuwen-ai/iCode) 的核心能力：
一个运行在终端里的编码智能体（coding agent）——TUI 对话、模型接入、流式输出、
工具调用与审批、会话持久化，以及无界面的一次性任务执行。

> 本仓库当前是 **核心功能的 Rust 版本（v0.1）**，并非 iCode 的完整移植。
> 已移植与尚未移植的能力见 [docs/porting.md](docs/porting.md)。

## 特性

- **TUI 对话界面**：基于 ratatui，包含对话视图、输入框、状态栏（模型 / 审批模式 /
  工作目录 / token / 会话），支持流式 Markdown 渲染、工具调用卡片与审批弹窗。
- **模型接入**：任何兼容 OpenAI Chat Completions 协议的服务（默认 DeepSeek），
  支持流式 SSE、`reasoning_content` 推理内容、token 用量统计、429/5xx 自动重试。
- **智能体循环**：系统提示词 + 多轮工具调用；只读检索批次并行执行并按调用顺序回填，
  写操作与交互工具保持顺序执行；内置 Code 与只读 Q&A 两种智能体。
- **工具集**：`read_file`、`write_file`、`edit_file`、`list_dir`、`grep`、`glob`、
  `bash`、`todo_write`、`ask_user`。
- **审批策略**：`ask`（默认，逐次确认写操作与 Shell）、`auto`、`deny`。
- **会话**：JSONL 追加写入，支持 `--continue`、`/sessions`、`/resume <id>`。
- **无界面运行**：`icode run "..."`，支持 `--json` 供脚本消费。
- **输入便捷**：`/` 命令、`!` 直接执行 Shell、`@file` 附加文件内容。

## 界面

真实终端截图（118×40，DeepSeek 兼容的 mock 模型流式输出）：

![TUI 对话：流式回答、Markdown 渲染与工具调用卡片](docs/screenshots/tui-answer.png)

![执行写操作前的审批弹窗](docs/screenshots/tui-approval.png)

![批准后工具执行结果与最终回答](docs/screenshots/tui-tool-result.png)

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
`/sessions`、`/resume <id>`、`/cwd <path>`、`!<shell 命令>`、`@<文件>`。

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
