# Learning Assistant Agent

## Project Context

**目标**：构建一个 AI 驱动的开源课程助教 Agent，帮助中文母语学习者高效学习 UCB/Stanford/MIT 等顶级名校的 CS/AI/Math/Statistics/DS 开源课程。

**核心痛点**：顶级开源课程数理性强、概念繁多、抽象性突出，英文资料阅读效率低于母语中文。

**目标用户**：具有英文阅读能力（能正常阅读讲义和资料，但速度和理解效率不如母语）的中文学习者。

## Tech Stack

| 层级 | 技术 | 用途 |
|------|------|------|
| Agent Framework | LangGraph | 有状态、多分支教学流程编排 |
| RAG/Indexing | LlamaIndex | 文档索引、混合检索 |
| Chat UI | Chainlit | 流式聊天、文件上传、会话管理 |
| LLM | Anthropic Claude API | 核心推理与生成 |
| Embedding | SiliconFlow BGE-M3 (主) / Voyage AI / OpenAI | 文本向量化（无GPU，全部走API） |
| Vector Store | ChromaDB | 向量存储与相似搜索 |
| PDF Parsing | PyMuPDF (主) / MinerU (可选) | 文档解析 |
| Package Manager | uv | Python 依赖管理 |

## Key Design Decisions

1. **四段式教学输出**：Core Process → Examples → Math & Notation → Summary
2. **术语保留**：所有英文专业术语和 LaTeX 符号不翻译
3. **混合检索**：稠密向量（语义）+ BM25（LaTeX 精确匹配）
4. **无 GPU**：所有 Embedding 走 API，不依赖本地模型
5. **Windows 兼容**：MinerU 作为可选依赖，PyMuPDF 为可靠回退

## Project Structure

```
learning_assistant_agent/
├── config/          # 配置与 Prompt 模板
├── src/
│   ├── ingestion/   # PDF 解析与分块
│   ├── indexing/    # Embedding、向量库、混合检索
│   ├── agent/       # LangGraph 图与节点
│   ├── teaching/    # 教学法模块
│   └── ui/          # Chainlit UI
├── tests/           # 端到端 + 单元测试（含 e2e_verify.py）
└── data/            # 文档、向量库、中间结果
```

## Development Phases

1. ✅ 环境搭建与项目骨架
2. ✅ 文档处理管道（PDF → Chunking → Embedding → ChromaDB）
3. ✅ 核心 RAG + 基础 Chainlit
4. ✅ LangGraph 智能体（四段式教学）
5. ✅ 多 Agent 团队验收 + 全量代码审计 + Bug 修复
6. ✅ 历史记录 + 上传去重 + Chainlit UI 修复
7. ✅ 检索精准度（section 元数据 / 文件名过滤 / 跨源多样性）+ 展示层去重 + 路由兜底

## Lessons Learned

> 以下教训来自本项目 Phase 1-5 中遇到的 15+ 个 Bug 的系统性复盘。

### 1. API 版本兼容性：永远不要假设，必须实际验证

**核心原则**：任何第三方 API 调用，必须在**目标版本的已安装包**上验证实际签名。

| 错误假设 | 实际行为 | 教训 |
|---------|---------|------|
| `msg.update(content="...")` 接受参数 | Chainlit v2.11.1 的 `update()` 无参数，需先 `msg.content = ...` 再 `await msg.update()` | **读源码确认每个方法签名**，不要基于其他框架的惯例推测 |
| `@cl.on_upload` 存在 | Chainlit v2.11.1 无此装饰器 | 检查 `dir(chainlit)` 或源码中的 `on_*` 列表 |
| LlamaIndex `OpenAIEmbedding` 用 `base_url` | 实际参数名是 `api_base` | API 兼容层（OpenAI SDK → LlamaIndex）的参数名可能不同 |
| `ChatAnthropic` 可以只传 `SystemMessage` | DeepSeek 的 Anthropic 兼容端点拒绝纯 system 消息（`messages` 数组为空） | **API 代理/兼容层有行为差异**，原生 Anthropic 行为 ≠ DeepSeek 行为 |

### 2. 数据流完整性：每一步的输出必须满足下一步的输入要求

| Bug | 断裂点 | 教训 |
|-----|-------|------|
| `embedding not set` | `chunker.chunk()` → `vector_store.add_nodes()` 之间缺少 embedding 步骤 | 管线每个环节都要验证：**上一步的输出类型是否匹配下一步的输入要求** |
| `'dict' object has no attribute 'relevance_score'` | `RetrievedContext` 是 TypedDict（运行时就是 dict），代码却用了 `.relevance_score` 属性访问 | Python TypedDict 只是类型提示，运行时是普通 dict，只能用 `['key']` 访问 |

### 3. 测试分层：静态审查无法替代端到端调用

| 测试层级 | 覆盖范围 | 遗漏类型 |
|---------|---------|---------|
| `import` 所有模块 | 编译期语法正确 | 运行时数据流 Bug |
| AST 静态分析 / grep 模式匹配 | 代码结构和签名 | API 行为差异、数据格式不匹配 |
| 单节点隔离调用 | 单个函数的逻辑 | **跨节点的数据传递**（chunk → embed → store, retrieval → explanation） |
| **`agent.ateach()` 全链路调用** | 完整管线 | **（无遗漏——这是最终验证标准）** |

**结论**：每次改动后，必须跑至少一次 `agent.ateach("测试问题")` 全链路验证。不要只做 import 检查或单节点测试。

### 4. DeepSeek API 特定问题

| 问题 | 说明 |
|------|------|
| `SystemMessage` 不能单独出现 | 每条消息列表必须至少包含一个 `HumanMessage` 或 `AIMessage`。纯 `[SystemMessage(...)]` 会被 langchain-anthropic 提取到 `system` 字段，`messages` 数组变空 |
| 响应格式非纯文本 | DeepSeek v4 返回 `[{type:'thinking',...}, {type:'text',text:'...'}]` 格式，需用 `extract_text()` 统一提取 |
| `base_url` 用 `/anthropic` 后缀 | DeepSeek 的 Anthropic 兼容端点 URL 是 `https://api.deepseek.com/anthropic` |

### 5. Chainlit v2.11.1 特定问题

| 问题 | 说明 |
|------|------|
| 无 `@cl.on_upload` / `@cl.on_file_upload` | 文件上传只能通过 `cl.AskFileMessage`（弹窗式）或 `message.elements`（消息附件）|
| `Message.update()` 无参数 | 更新消息内容需：`msg.content = "new"; await msg.update()` |
| 每个 `@cl.on_*` 装饰器只能注册一次 | 多个文件中重复的 `@cl.on_chat_start` 只会保留最后一个 |
| LaTeX 渲染需在 `config.toml` 中启用 | `[features] latex = true` |

### 6. 代码设计原则

| 原则 | 反例 | 正例 |
|------|------|------|
| 不访问私有属性 | `retriever._bm25._initialized` | 在 `BM25Retriever` 上添加 `is_initialized()` 公共方法 |
| 跨模块用公共 API | `hr_module._retriever = ...` | 提供 `set_retriever()` 函数 |
| 单例无需锁（asyncio） | 添加不必要的 `asyncio.Lock` | 同步无 `await` 的 getter 在 asyncio 中天然原子 |
| Prompt 去重 | `SystemMessage` + `HumanMessage` 中重复嵌入同一文本 | System prompt 只在 `SystemMessage` 中传递 |

### 7. Chainlit v2.11.1 真实 API 签名必须读源码确认

**核心原则**：`AskActionMessage.send()` 直接返回 `AskActionResponse`（不是 `wait_for_answer()`），`wait_for_answer` 是 **bool 属性不是方法**（调用会抛 `TypeError: 'bool' object is not callable`）；`AskFileMessage` 是**阻塞弹窗**会锁住输入框，不能用于"打开对话即可输 `/history`"场景；`VectorStoreQuery.filters` 要 `MetadataFilters` 对象不是 dict。**永远不假设框架 API 与惯例一致**，一律查 `.venv/Lib/site-packages/<pkg>/` 源码。

| 问题 | 教训 |
|------|------|
| 调用 `msg.wait_for_answer()` 报错 | 读源码：`send()` 返回 `AskActionResponse`，`wait_for_answer` 是 bool 字段 |
| 打开对话必须先上传才能输指令 | `AskFileMessage` 阻塞输入框；改为非阻塞的 `message.elements` + 文字提示 |
| `[features.spontaneous_file_upload]` 缺失 | `.chainlit/config.toml` 必须显式 `enabled = true`，否则输入框 📎 按钮**不显示** |
| `filters={"source": x}` 报错 | LlamaIndex 要 `MetadataFilters(filters=[MetadataFilter(...)])` 对象 |

### 8. `str.format()` 与 LaTeX 花括号冲突

**核心原则**：任何被注入 prompt 模板的内容（检索 chunk / LLM 输出）可能含 LaTeX 下标 `X_{ij}`、`beta_{t-1}`，`str.format()` 会把 `{ij}` 当成格式字段抛 `KeyError: 'ij'`。

| 错误做法 | 正确做法 |
|---------|---------|
| `PROMPT.format(retrieved_context=chunk)` | `render_prompt(PROMPT, retrieved_context=chunk)`（src/utils.py） |

`render_prompt` 用正则只替换声明的 `{placeholder}`，注入值不再被二次解析。所有 6 个节点统一走 `render_prompt`。

### 9. 路由不可单字信任 LLM 判断

**核心原则**：`intent`/`needs_math`/`needs_example` 由 LLM 生成易误判。代码层必须有兜底，否则教学流程被误截断。

| 误判案例 | 兜底措施 |
|---------|---------|
| "开始讲解"被判 `navigate` → 直送 END 显示空问候 | `query_understanding_node` 检测教学意图关键词命中且非纯问候 → 降级 `learn_new` |
| softmax / loss / gradient 主题被判 `needs_math=False` → 跳过 ③ Math 节点（用户输出②后直接④） | `query_understanding_node` 检测数学信号词命中 → 强制 `needs_math=True`；`_route_after_example` 默认走 `math_notation`，仅 `intent ∈ {refuse, navigate, review}` 才跳 |

**永远不要用一个 LLM 字段决定流程跳过**——必须有第二条独立信号或代码层默认值兜底。

### 10. 检索索引会被"无去重的反复 ingest"污染

**核心原则**：dedup 功能之前的多次 ingest 把同一份 PDF 反复入库 9 遍，81 chunk 中只有 9 个唯一，72 个重复让 BM25/dense 排名被同一 chunk 垄断，top-5 实际只是 1-2 个唯一 chunk —— 用户问"第2/3部分"时 LLM 完全无法区分。

| 防护层 | 实施 |
|-------|------|
| 上传层 SHA-256 去重 | `file_handler.process_uploaded_files` 算文件级 hash，命中全局 registry 即跳过 ingest |
| 索引层 `(source, content)` 查重 | `VectorStoreManager.add_nodes` 每次 add 前查重，未来再污染也能挡 |
| 一次清理存量 | `VectorStoreManager.dedup_by_content()`，retriever 构造时自动跑一次 |

### 11. 展示层去重必须"全文扫描"而非"前导扫描"

**核心原则**：LLM 常先写一句导语"好的，现在我们来写 Part Ⅱ…"（含大量正文词），再写裸标题行 `② Interspersed Examples（示例展开）`。只剥前导行的 stripper 见到首句导语就停止，**漏掉导语后的标题行** → 输出仍重复。

| 错误策略 | 正确策略 |
|---------|---------|
| 扫描到首个非 header 行就停止 | 遍历**所有**行，删除任何"短、标题-dominated、不在数学块内、非 Example/Step 编号列表项"的独立 header 行 |
| 仅检查首行 | 跳过 `$...$`/`$$...$$` 数学块内、保留含正文词 ≥ 2 的导语行 |

`_strip_duplicate_part_header`（methodology.py）现在全文扫描，每行独立判定。

### 12. chunk 必须带 section 元数据

**核心原则**：用户问"讲讲 xxx.pdf 第2部分和第3部分"时，chunk 无 section 标签 → LLM 不知哪段是 part 2 / part 3，把 "Evaluation" 误说成 "实际上在笔记里是第二部分"。

| 实施 | 说明 |
|------|------|
| 检测 3 种 heading | markdown `## 2 T`、同行 `2 T`、PDF 跨行 `2\nT` |
| Title-Case 启发式过滤列表项 | 末词须大写字母开头（Vectors/Matrix/Tasks），过滤 `1 Gather fixed size…`、`1 Course Instructors` |
| 写 `metadata['section_heading']` + 文本前缀 `[Section: N Title]` | dense embedder / BM25 / teaching LLM 三者都能看到 section 上下文 |
| 文档顺序传播最后一个 heading | 不含自己 heading 的 chunk 继承前一 chunk 的 section |

### 13. 每次功能更新必须调用 subagent 执行端到端验证

**核心原则**：`tests/e2e_verify.py` 涵盖 S1 开对话 → S2 上传 + dedup → S3 提问 `assemble_full_response` 标题去重 → S4 路由 `needs_math` 兜底 → S5 检索 source 过滤 + 跨源/跨 section 多样性五阶段。**提交 commit 前必须用 `task` 工具调用一个 `explore` 类型 subagent 跑此脚本并报告 PASS/FAIL**；FAIL 一律补修复，**禁止凭 import OK 或单节点测试通过就提交**。

## Testing Checklist

每次代码改动后，按以下顺序验证：

```
1. uv run python -c "from src.agent.graph import TeachingAgent"     # 导入链
2. uv run python -c "from src.ingestion.pipeline import IngestionPipeline; p = IngestionPipeline()"  # 组件初始化
3. uv run python -c "agent = TeachingAgent(); asyncio.run(agent.ateach('test question', 'test'))"  # 全链路
4. uv run python tests/e2e_verify.py                                  # 端到端 S1-S5（提交前必过）
```

第 4 步是**提交前硬性门槛**：覆盖历史记录 / 上传去重 / 标题去重 / 路由兜底 / 检索多样性。任何一步 FAIL 不得提交。
