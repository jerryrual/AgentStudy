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
| 上下文裁剪（Phase 1 / 2 / 3 + 轮次后强制折叠） | 请求输入跨过派生触发阈值后，Phase 1 按从旧到新的顺序将已完成旧轮次的工具结果替换为确定性摘要；仅当仍超过目标预算时，Phase 2 才按从旧到新的顺序移除已完成旧轮次的工具调用/结果批次；若此后仍超过目标预算，Phase 3 才按从旧到新的顺序折叠已完成的历史轮次。Phase 1 / 2 仍仅作用于当次请求；Phase 3 批次会在 provider 调用前原子追加到 JSONL，失败则不调用 provider。严格的 provider 上下文溢出会绕过常规触发阈值，强制运行现有 Phase 1–3，并仅在保守估算的输入上下文严格缩小时立即重试一次；provider 明示更小窗口、无法继续缩小或第二次仍溢出时直接返回错误。成功轮次结束后，若最终一次模型调用报告的输入 token 严格超过派生 force 阈值，仍会持久化折叠一个最旧的先前已完成轮次，同时保留刚完成轮次和所有较新历史；原始 JSONL 记录仍为追加式审计历史，TUI 可见记录不因请求时折叠而改变 | `src/context_trim.rs`、`src/compaction.rs`、`src/model/client.rs` |
| 工具调用 | 多轮工具循环；纯只读工具批次并行执行，结果按调用顺序回填；超长输出截断 | `src/agent/` |
| 文件读写 | `read_file`、`write_file`、`edit_file`（精确匹配替换） | `src/tools/fs.rs` |
| 代码搜索 | `grep`（正则）、`glob`、`list_dir`，遵循 .gitignore | `src/tools/search.rs` |
| Shell | `bash` 工具，超时与 `kill_on_drop` | `src/tools/shell.rs` |
| 审批 | ask / auto / deny，写操作与 Shell 需确认 | `src/interaction.rs` |
| 计划清单 | `todo_write` | `src/tools/misc.rs` |
| 提问 | `ask_user`（TUI 中等待回答） | `src/tools/misc.rs` |
| 会话 | JSONL 追加写入、列举、继续；`/compact` 生成并持久化完整续写摘要；请求时 Phase 3 以原子折叠批次持久化，轮次后 force 折叠以边界计数检查点持久化，并在 `/resume` 时保留未折叠尾部；Phase 4 用单条 `ContextProjection` 同时保存历史折叠与经严格边界校验的当前轮次 Last Words 替换，TUI 以 FIFO 屏障保证同轮消息先落盘；Phase 4 breaker 以版本化 sidecar 快照绑定 user-turn ordinal，不向主 JSONL 写入新记录类型，并兼容读取/迁移 PR #25 曾写入 transcript 的快照；每次 side-call 在 provider I/O 前先追加累计 token 状态，跨 `reopen` 句柄/进程通过会话文件锁合并 spend，写入失败或旧轮 gate 会拒绝调用，新 User 自动隔离旧状态；每个平铺的 `<id>.jsonl` 旁保留稳定的 `<id>/` 归档根，resume 时在不改写 JSONL 的前提下 reconciliation；TUI 取消后若暂时无法重载权威会话历史，会阻止新的模型轮次，直到重载成功或切换会话 | `src/session.rs`、`src/phase4_breaker.rs`、`src/compaction.rs`、`src/tui/app.rs` |
| Phase 4 归档与检索 | 已实现上游目录/文件名约束、2 MiB 单记录与 64 MiB 单轮上限、256 MiB 会话配额与最旧记录淘汰、owner-only exclusive 写入、append-only `catalog.jsonl`/tombstone、可重建 `manifest.md`、严格 UTF-8 中段截断，以及 path/symlink/junction containment。当前轮次投影会把每个 assistant/tool 组完整未脱敏地序列化（并行调用共用一条记录，保留 assistant 文本、reasoning、provider-native items、完整参数和结果），Run/TUI 在同一 session append/FIFO 屏障内先归档再提交 `ContextProjection`；归档 I/O、限额或安全校验失败时拒绝投影并保留原始工具历史。成功归档的同轮记录会累积为 Last Words manifest（状态保留最新 500 行、请求渲染不超过 50 行/6000 字符，配额淘汰后标记 missing），进入紧随其后的 Chat/Responses 主请求及后续同轮 Last Words side-call；新用户轮次会清空该 manifest。每个逻辑轮次开始时还会冻结此前 live 记录数量，并在 `read_file` 可用时把经过安全校验的绝对 `catalog.jsonl` 路径注入该轮所有 Chat/Responses 请求；模型可用通用 `read_file` 读取该绝对 catalog，再按其中的 session-relative path 读取完整归档记录；同轮新增归档不会改变该指针 | `src/compaction_archive.rs`、`src/session.rs`、`src/model/client.rs`、`src/run.rs`、`src/tui/app.rs`、`tests/tools.rs` |
| 项目记忆 | 主 agent 从当前工作目录加载 `AGENTS.md` 并注入系统上下文 | `src/memory.rs` |
| 无界面运行 | `icode run`，JSON 输出 | `src/run.rs` |
| 直接 Shell | 输入 `!命令` 直接执行 | `src/tui/app.rs` |
| 深度推理 | 展示 `reasoning_content` 摘要 | `src/tui/ui.rs` |
| 模型协议 | Chat Completions；核心无状态 Responses API（文本、推理摘要、本地函数工具、流式/非流式） | `src/model/` |
| 重试 | 429/5xx/transport 指数退避重试；严格上下文溢出会强制运行 Phase 1–3，并在当前工具尾部可严格缩小时生成无工具、非流式 Last Words，原子持久化 Phase 4 后零延迟重试一次；全部恢复仍属于同一 agent round。`phase4_side_call_token_budget` 按逻辑用户轮累计每次实际 Phase 4 HTTP 尝试的估算输入 token：`-1` 不限、`0` 禁用、正数按上游的 charge-first 且严格小于上限准入；格式纠正和 429/5xx/transport 重试共享配额，新用户轮重置 | `src/config.rs`、`src/agent/events.rs`、`src/model/client.rs` |

## 尚未移植

以下按依赖顺序推进；各项列出尚未实现的部分。

### 1. 上下文 / 协议

1. 其余 Phase 4 接线。Phase 1–4 已能按预算或 provider 溢出触发：Last Words side-call 不流式且不暴露工具，校验新 note 后将历史折叠和完整当前工具尾部替换原子持久化，严格缩小请求才在同一 agent round 继续；失败或取消不会丢弃工具历史，后续工具轮可再次合并旧 note 与新增工作。被投影替换的 assistant/tool 组已经按上游格式完整未脱敏归档（并行调用共用记录），Run/TUI 在 FIFO/session append 屏障内先写 record/catalog/manifest 再提交投影，任何归档拒绝或错误都会保留 raw JSONL。成功提交的同轮归档 rows 会跨多次 Phase 4 累积为有界 Last Words manifest，紧随当前提交的主请求会得到新 rows，后续 side-call 也能保留此前 rows；新用户轮次清空状态。此前轮次的 live catalog 数量和经过校验的绝对路径会在逻辑轮次开始时冻结，并在 `read_file` 可用时注入该轮所有请求；通用 `read_file` 可沿 catalog 中的相对路径检索完整压缩记录。breaker 已在任何 Phase 4 planning/provider 工作前 write-ahead 持久化 attempts，并在第 100 次后或 side-call 配额已耗尽时持久化 disabled；成功提交后会按最终 archive manifest 请求计算绝对 context utilization 降幅，低于 5 个百分点持久化 no-progress 与 sticky tail_override，连续第二次即 disabled，达到阈值则只重置连续计数；Last Words generation failure 会持久化普通 strike，side-call budget 拒绝会持久化 strike 并立即 disabled，任务取消不会经过这两个返回路径；projection archive/persistence 拒绝或异常会在保留 raw groups 时持久化 spill-abort strike，TUI cancellation gate 则明确不记 strike；headless `resume-turn` 与 TUI `/continue-turn` 可显式从完整持久化的工具批次或 Last Words checkpoint 恢复同一 user-turn 及其 breaker，且不会追加新 User；显式打开或 `/resume` 的既有 TUI 会话会先只读验证相同恢复边界，在不采用 sidecar 或改写 transcript 的前提下进入 Continue prompt，空 Enter 复用 `/continue-turn`，取消/失败后重新提示，新 turn 或 `/new` 清除提示，checkpoint/breaker 读取或解析失败会显示错误并阻止原会话启动新 turn；仅有 User 或持久化记录可识别的中断工具组会 fail closed；较早 checkpoint 无法证明其后没有未落盘的工具副作用，因此 headless 恢复会继续拒绝全部变更型工具，TUI 恢复则在 `deny` 模式拒绝、在 `ask`/`auto` 模式对每次变更调用强制单次确认（不会继承或新增 always-allow）。每个 headless run 与打开会话的 TUI 都会在 load/reopen 前非阻塞取得独立 `.run.lock` 并持续持有，阻止跨进程并发执行同一会话；TUI `/resume` 与 `/new` 只在新 session+lease 全部就绪后原子切换，失败保留原会话。Run/TUI 已在每次主 provider 请求前同步原子写入 owner-only `session.recovery.json`，快照包含完整模型消息、当前 turn、breaker 以及已同步 transcript 的因果位置；持久化失败会阻止请求，干净且 transcript 已落盘的完成会删除 sidecar，错误、取消或未保存消息则保留。headless `resume-turn` 与 TUI `/continue-turn` 会在 session/run lease 内显式采用仍绑定同一 transcript 位置且消息相同或包含完整可持久化后缀的较新 v2 sidecar，先原子补写后缀；headless 继续拒绝变更型工具，TUI 按上述模式拒绝或逐次确认；普通启动、`run --session`、TUI `/resume` 仍只读取主 transcript，缺少因果位置的 v1 sidecar 不会采用。普通本地工具组的 sidecar 尾部若已有至少一个完整结果，显式恢复会按 provider 调用顺序只保留已配对的 call/result、保留 assistant 可见文本，并在采用前重新执行完整边界与原子后缀校验；零结果组、原生 Responses `function_call` item 缺少 `call_id`、与 assistant 调用不匹配、重复或缺失的组，以及孤儿/重复/错名结果仍 fail closed。Agent 会在每个本地工具结果（含错误结果）完成后，把不可变消息前缀与按 provider 调用顺序排列的已完成子集提交给 turn-scoped 单线程 writer；连续待写快照按 latest-wins 合并，主 provider 前的 exact barrier 同步等待同一顺序域，成功清理与失败/取消 preserve 均先封闭 intake 并 drain，迟到的 partial snapshot 也不能重绑定到较新的 transcript，因此旧后台写不能越过 barrier 或在删除后复活 sidecar；并行批次的 canonical Tool 与 JSONL 仍只在整批成功后按调用顺序提交，checkpoint 失败会终止该轮且不会继续 provider。部分原生 Responses 组会同步过滤未完成的 `function_call` item，并保留其他原生 item 的原顺序。同一 turn 的 `InjectedUser` 已有带稳定 ID 的独立消息/JSONL 类型，Chat/Responses 只发送标准 user role，恢复 sidecar 可采用该 same-turn 后缀，turn/breaker/trim/compaction 边界仍只由原始 User 定义，历史摘要与 Phase 4 原始归档会保留注入内容；Agent 已有单槽 reservation 队列，并只在完整工具组之间、每次逻辑 provider 调用前同步调用注入 transcript sink，再经过 strict checkpoint barrier；tool-free 完成与 enqueue 会原子线性化，provider 内部重试不会重复 drain。TUI 已为每个运行中轮次创建 `InjectionQueue`/`InjectionSource` 并调用 `Agent::run_with_injections`；`TuiSink` 会在此前 `MessageAdded` 之后同步排队注入 transcript 持久化事件，等待写入 JSONL，再由 Agent 跨过 strict recovery barrier 后调用 provider。TUI 可见输入切片已接线：运行中按 Enter 会锁定并排队单个草稿，状态栏显示 `queued`；首次 Esc 撤回 pending 输入，第二次才取消；注入内容持久化消费后渲染 user bubble，并仅清除 ID 匹配的草稿；ask 回答使用隔离的临时编辑器；输入若与已封闭的 terminal state 竞态，则会在前一状态安装后启动一个新的 turn。ACP ingress 与显式后端 receipt/event 协议仍未移植；当前实现每次逻辑模型调用最多进行一轮 Phase 4，并将 note 格式纠正限制为两次生成。side-call 输入 token 累计配额已使用上游兼容的版本化 breaker schema，按 user-turn ordinal 回放，并在每次物理 provider 尝试前 write-ahead 持久化；新 User 重置状态。普通 Rust resume 仍会先追加新 User 并重置 breaker；只有显式 `resume-turn` / `/continue-turn` 保留同一 turn。
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

1. ACP 服务（`icode acp` / `icode serve`）与编辑器集成。Agent 已通过向后兼容的富工具事件与审批钩子向协议适配器保留 provider 分配的 `toolCallId` 和原始参数，重复或并行同名工具可稳定关联；内部有序、非阻塞的 turn-event channel 已将同步 observer 回调桥接到异步消费边界，异步 dispatcher 现会排空有序 turn events，将其映射为官方稳定 ACP v1 `SessionNotification`，并按序准入同一 session 的 ACP connection 无界出站队列；准入成功不代表传输送达。进程内 ACP session registry 现会创建或打开持久化本地会话，直接复用其 ID 作为强类型 ACP session ID，在注册项存续期间持有既有的跨进程 run lease，每个会话一次只允许一个 prompt，并提供协作式取消；发出取消后仍保持 busy，直到 prompt guard 退出。空闲注册项可显式关闭并释放 lease，活动 prompt 存在时关闭会失败。ACP prompt 内容现已有纯适配器：按原顺序把基础必需的文本块和资源链接投影为现有的单条用户消息表示；空内容或当前不支持的图片、音频、内嵌资源会在会话持久化和 provider I/O 之前被拒绝。ACP session control 纯逻辑现会生成不夸大的稳定 v1 初始化响应，严格验证新会话 cwd，并在任何持久化前拒绝 MCP server 或 additional roots，成功时返回强类型 session ID；cancel 仅向匹配的活动 prompt 转发协作式信号。从持久化 `Record` 到有序官方稳定 ACP v1 历史 replay `SessionUpdate` 的纯投影现已实现并有测试覆盖，但尚未接入 load handler；在 replay dispatch 能事务性地、严格先于响应返回完成之前，load handler/capability 仍不可用。SDK callback handler 尚未接线；stdio 入口和 prompt handler 也尚未实现，因此不能暴露初始化成功的半成品 Agent。MCP/additional roots、权限请求、prompt 完成以及 usage/notice 处理也仍不可用。
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
