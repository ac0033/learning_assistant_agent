# AGENTS.md — 指导 opencode agent 协作时的核心约束

> 配套文档：`CLAUDE.md`（完整 Lessons Learned 13 条 + 项目背景）。本文件只列 opencode agent 上手必读的硬性约束与必跑命令。

## 项目简介

`learning_assistant_agent` 是一个 AI 驱动的开源课程助教 Agent，帮助中文母语学习者高效学习 UCB/Stanford/MIT 等顶级名校的 CS/AI/Math/Statistics/DS 开源课程。技术栈：LangGraph + LlamaIndex + Chainlit + Anthropic 兼容 API + SiliconFlow BGE-M3 + ChromaDB。

## 必跑命令

```bash
# 1. 导入链（最小验证，秒级）
uv run python -W error::SyntaxWarning -c "from src.agent.graph import TeachingAgent"

# 2. 组件初始化（不带 LLM 调用，秒级）
uv run python -c "from src.ingestion.pipeline import IngestionPipeline; p = IngestionPipeline()"

# 3. 端到端 5 阶段（提交前硬性门槛）
uv run python tests/e2e_verify.py

# 4. 启服务（本地调试用）
uv run chainlit run src/main.py --port 8000 --host 0.0.0.0
```

## 关键禁忌

1. **禁止用 `str.format()` 注入任何检索 chunk / LLM 输出** —— LaTeX 花括号 `X_{ij}` 会被当格式字段，抛 `KeyError: 'ij'`。一律用 `src/utils.render_prompt()`。
2. **禁止假设 LLM 单字段正确** —— `intent` / `needs_math` / `needs_example` 会误判。路由必须有代码层兜底（详见 `graph.py::_route_after_example` 与 `query_understanding_node` 的关键词兜底）。
3. **禁止假设 Chainlit / LlamaIndex API 与惯例一致** —— 必须查 `.venv/Lib/site-packages/<pkg>/` 源码确认方法签名。已知坑：`AskActionMessage.wait_for_answer` 是 bool 属性不是方法；`VectorStoreQuery.filters` 要 `MetadataFilters` 对象不是 dict。
4. **禁止凭 import OK 或单节点测试通过就提交** —— 提交前必须跑 `tests/e2e_verify.py` 全 PASS。
5. **禁止无去重地反复 ingest 同一 PDF** —— 会污染索引（曾出现 81/9 重复）。上传走 `file_handler.process_uploaded_files`（SHA-256 dedup），更深一层 `VectorStoreManager.add_nodes` 还有 `(source, content)` 查重兜底。
6. **禁止翻译核心英文术语 / LaTeX 符号** —— 这是产品定位（术语保留让用户对照原文）。
7. **禁止单字段决定教学流程跳过** —— `_route_after_example` 默认走 `math_notation`，仅 `intent ∈ {refuse, navigate, review}` 才跳过数学节点。

## 回退指引

- `checkpoint-before-optimization` tag：Phase 1-6 完成后的稳定基线（HEAD `bdf1d13`）。
- `checkpoint-after-fix` tag：本轮（标题去重全文扫描 + 路由 needs_math 兜底 + e2e_verify）完成后的稳定基线。
- 回退命令：`git reset --hard <tag>`。也可逐 commit 单独 revert。

## 常用命令清单

```bash
# 查看现有索引
uv run python -c "from src.indexing.vector_store import VectorStoreManager; vsm=VectorStoreManager(); print('total:', vsm.count()); print('sources:', vsm.list_sources())"

# 一次性清理存量重复 chunk（retriever 构造时也会自动跑一次）
uv run python -c "from src.indexing.vector_store import VectorStoreManager; vsm=VectorStoreManager(); print('removed:', vsm.dedup_by_content())"

# 删某文件的 chunks 后重新 ingest（让某文件享受新 chunker 的 section 元数据）
uv run python -c "from src.indexing.vector_store import VectorStoreManager; from src.ingestion.pipeline import IngestionPipeline; from pathlib import Path; vsm=VectorStoreManager(); vsm.delete_by_source('xxx.pdf'); IngestionPipeline().ingest_file(Path('data/documents/xxx.pdf'))"

# 手工跑一次教学全链路（不依赖 UI）
uv run python -c "import asyncio; from src.agent.graph import TeachingAgent; a=TeachingAgent(); r=asyncio.run(a.ateach('Explain softmax in word vectors', 'test_t')); print({k: (v[:80] if isinstance(v,str) else v) for k,v in r.items()})"
```

## 改动后核查清单（每次功能更新必跑）

1. `uv run python -W error::SyntaxWarning -c "from src.agent.graph import TeachingAgent"`   # 导入链
2. `uv run python tests/e2e_verify.py`                                                       # 端到端 5 阶段
3. **调用 subagent 跑同一脚本并使其报告 PASS/FAIL**                                         # 强制第三方复核
4. FAIL 不得提交；PASS 之外仍需做一次 `agent.ateach("Explain softmax…")` 全链路小测

**核心约束：禁止仅凭 import OK 或单节点测试通过就提交。** 端到端验证是硬性门槛。Step 3 的 subagent 调用必须用 `task` 工具，`subagent_type="explore"`，提示词明确要求它执行 `uv run python tests/e2e_verify.py` 并逐阶段报告 PASS/FAIL。