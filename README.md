# 🎓 开源课程助教 Agent / Learning Assistant Agent

> 帮助中文母语学习者高效学习 UCB / Stanford / MIT 等顶级名校开源课程（CS / AI / Math / Statistics / Data Science）的 AI 助教 Agent。
>
> An AI teaching assistant that helps Chinese-speaking learners study top university open-source courses, explaining complex concepts in Chinese while preserving all English technical terms and LaTeX notation.

## ✨ 核心特性 / Features

- **四段式教学输出 / Four-Part Teaching**：Core Process → Examples → Math & Notation → Summary
- **混合检索 / Hybrid RAG**：稠密向量（语义）+ BM25（LaTeX 符号精确匹配）
- **术语保留 / Terminology Preservation**：英文专业术语与 LaTeX 符号不翻译，便于对照原文
- **流式输出 / Streaming Output**：基于 Chainlit WebSocket 的实时 token 流
- **会话记忆 / Session Memory**：LangGraph MemorySaver 支持跨轮对话上下文
- **上传去重 / Upload Dedup**：SHA-256 文件级 + (source, content) chunk 级双重去重
- **多 LLM / Embedding 后端**：支持 Anthropic / DeepSeek / OpenRouter；SiliconFlow BGE-M3 / OpenAI / Voyage AI

## 🧱 技术栈 / Tech Stack

| 层级 | 技术 | 用途 |
|------|------|------|
| Agent Framework | LangGraph | 有状态多分支教学流程编排 |
| RAG / Indexing | LlamaIndex | 文档索引与混合检索 |
| Chat UI | Chainlit | 流式聊天 / 文件上传 / 会话管理 |
| LLM | Anthropic 兼容 API | 核心推理（Claude / DeepSeek / OpenRouter）|
| Embedding | SiliconFlow BGE-M3 (默认) / OpenAI / Voyage | 文本向量化，无 GPU |
| Vector Store | ChromaDB | 向量存储与相似搜索 |
| PDF Parsing | PyMuPDF (主) / MinerU (可选) | 文档解析 |
| Package Manager | uv | Python 依赖管理 |

## 📦 安装 / Installation

### 前置要求 / Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) 包管理器
- 一个 LLM API Key（Anthropic / DeepSeek / OpenRouter 任选其一）
- 一个 Embedding API Key（推荐 SiliconFlow，有免费额度，支持微信/支付宝）

### 步骤 / Steps

```bash
# 1. 克隆仓库
git clone https://github.com/ac0033/learning_assistant_agent.git
cd learning_assistant_agent

# 2. 安装依赖（uv 会自动创建 .venv 并锁定版本）
uv sync

# 3. 配置环境变量
cp .env.example .env
#   编辑 .env，至少填写：
#     ANTHROPIC_API_KEY=...        # LLM key
#     ANTHROPIC_BASE_URL=...       # DeepSeek/OpenRouter 用户填对应地址
#     MODEL_ID=...                 # 如 deepseek-v4-flash / claude-sonnet-4-6
#     EMBEDDING_PROVIDER=siliconflow
#     SILICONFLOW_API_KEY=...
```

## 🚀 运行 / Usage

### 启动 Web UI（推荐）

```bash
uv run chainlit run src/main.py --port 8000 --host 0.0.0.0
```

浏览器打开 `http://localhost:8000`，拖拽上传课程 PDF，提问即可获得四段式讲解。

### 命令行使用 / CLI

本项目核心交互在 Chainlit Web UI，但也提供命令行入口用于脚本化测试：

```bash
# 一次性跑完整教学链路（不依赖 UI）
uv run python -c "import asyncio; from src.agent.graph import TeachingAgent; a=TeachingAgent(); r=asyncio.run(a.ateach('Explain softmax in word vectors', 'test_t')); print(r)"

# 端到端 5 阶段验证（提交前硬性门槛）
uv run python tests/e2e_verify.py
```

### 索引管理常用命令

```bash
# 查看现有索引
uv run python -c "from src.indexing.vector_store import VectorStoreManager; vsm=VectorStoreManager(); print('total:', vsm.count()); print('sources:', vsm.list_sources())"

# 清理存量重复 chunk
uv run python -c "from src.indexing.vector_store import VectorStoreManager; vsm=VectorStoreManager(); print('removed:', vsm.dedup_by_content())"
```

## 📂 项目结构 / Project Structure

```
learning_assistant_agent/
├── config/            # 配置与 Prompt 模板（Pydantic Settings）
├── src/
│   ├── main.py        # Chainlit 入口
│   ├── ingestion/     # PDF 解析与分块
│   ├── indexing/      # Embedding、向量库、混合检索
│   ├── agent/         # LangGraph 图与节点（四段式教学）
│   ├── teaching/      # 教学法模块
│   └── ui/            # Chainlit UI 层（会话/历史/上传/聊天）
├── tests/             # 端到端验证 e2e_verify.py
└── data/              # 文档、向量库、中间结果（gitignore，仅保留 .gitkeep）
```

## ✅ 开发阶段 / Roadmap

- ✅ Phase 1：环境搭建与项目骨架
- ✅ Phase 2：文档处理管道（PDF → Chunking → Embedding → ChromaDB）
- ✅ Phase 3：核心 RAG + Chainlit 集成
- ✅ Phase 4：LangGraph 智能体（四段式教学）
- ✅ Phase 5：多 Agent 团队验收 + 全量代码审计 + Bug 修复
- ✅ Phase 6：历史记录 + 上传去重 + UI 修复
- ✅ Phase 7：检索精准度（section 元数据 / 文件名过滤 / 跨源多样性）+ 路由兜底

## 🔒 安全提示 / Security

- `.env` 已在 `.gitignore` 中，**切勿提交真实 API Key**。
- `data/` 下文档与向量库均不入库，仅保留 `.gitkeep` 占位。
- 上传 PDF 走 SHA-256 去重，避免重复索引污染。

## 📜 License

MIT — 见 [LICENSE](LICENSE)
