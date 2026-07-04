# 🎓 开源课程助教 Agent

AI-powered teaching assistant for self-studying top university open-source courses (UCB, Stanford, MIT) in CS, AI, Math, Statistics, and Data Science.

## Overview

This agent acts as a **senior MIT/Stanford TA**, using the **Feynman Technique** to explain complex concepts intuitively in Chinese while preserving all English technical terms and mathematical notation. It processes your uploaded course materials (notes, slides, textbooks) and delivers structured, four-part explanations.

### Four-Part Teaching Structure

| Part | Content | Description |
|------|---------|-------------|
| ① | **Core Process** | Clear, logical walkthrough of the original content |
| ② | **Interspersed Examples** | Concrete, detailed examples with step-by-step logic |
| ③ | **Math & Notation** | Rigorous formula derivation, symbol-by-symbol explanation |
| ④ | **Summary** | Core takeaways distilled, bridge to related topics |

### Key Features

- 🧠 **Hybrid RAG**: Dense vector + BM25 retrieval (critical for LaTeX symbol exact matching)
- 📊 **LaTeX-Aware PDF Parsing**: PyMuPDF for reliable extraction, MinerU support for math-heavy PDFs
- 🔄 **Streaming Output**: Real-time token-by-token response via Chainlit WebSocket
- 📝 **Session Persistence**: LangGraph MemorySaver for cross-session conversation memory
- 🔤 **Terminology Preservation**: All technical terms kept in English, explained in Chinese

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph |
| RAG / Indexing | LlamaIndex |
| Chat UI | Chainlit |
| LLM | Anthropic Claude API |
| Embedding | Voyage AI / OpenAI |
| Vector Store | ChromaDB |
| PDF Parsing | PyMuPDF (primary), MinerU (optional) |
| Package Manager | uv |

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key
- Voyage AI or OpenAI API key (for embeddings)

### Setup

```bash
# Clone and enter the project
cd learning_assistant_agent

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your actual API keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   VOYAGE_API_KEY=pa-...        (or OPENAI_API_KEY=sk-...)
```

### Run

```bash
# Start the Chainlit app
uv run chainlit run src/main.py
```

The app will start at `http://localhost:8000`.

### Usage

1. **Upload course PDFs** — Drag & drop or click the 📎 button
2. **Ask questions** — "Explain gradient descent" / "什么是 eigenvalue decomposition?"
3. **Get structured explanations** — The agent responds with the four-part format

## Project Structure

```
learning_assistant_agent/
├── config/
│   ├── settings.py          # Centralized configuration (Pydantic)
│   └── prompts.py           # Teaching system prompts & templates
├── src/
│   ├── main.py              # Chainlit entry point
│   ├── ingestion/           # PDF loading & chunking
│   ├── indexing/            # Vector store & retrieval
│   ├── agent/               # LangGraph teaching agent
│   ├── teaching/            # Teaching methodology
│   └── ui/                  # Chainlit UI layer
└── data/                    # Documents, ChromaDB, processed files
```

## Development Phases

- ✅ **Phase 1**: Environment setup & project skeleton
- ✅ **Phase 2**: Document processing pipeline
- ✅ **Phase 3**: Core RAG + Chainlit integration
- ✅ **Phase 4**: LangGraph agent (four-part teaching)
- ⬜ **Phase 5**: Multi-agent team validation

## License

MIT
