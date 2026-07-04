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
└── data/            # 文档、向量库、中间结果
```

## Development Phases

1. ✅ 环境搭建与项目骨架
2. ✅ 文档处理管道（PDF → Chunking → Embedding → ChromaDB）
3. ✅ 核心 RAG + 基础 Chainlit
4. ✅ LangGraph 智能体（四段式教学）
5. ✅ 多 Agent 团队验收 + 全量代码审计 + Bug 修复

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

## Testing Checklist

每次代码改动后，按以下顺序验证：

```
1. uv run python -c "from src.agent.graph import TeachingAgent"     # 导入链
2. uv run python -c "from src.ingestion.pipeline import IngestionPipeline; p = IngestionPipeline()"  # 组件初始化
3. uv run python -c "agent = TeachingAgent(); asyncio.run(agent.ateach('test question', 'test'))"  # 全链路
```
