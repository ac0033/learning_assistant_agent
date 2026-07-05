"""Chainlit chat message handler.

Orchestrates the teaching flow:
  User message → LangGraph agent → streaming response → Chainlit UI

Key behaviors:
- Streaming: tokens appear in real-time via astream_events
- Four-part output: sections are rendered as they complete
- LaTeX: math notation is preserved for Chainlit's LaTeX renderer
- Sessions: per-user conversation state via SessionManager
- File uploads: initial upload via cl.AskFileMessage, subsequent via message.elements
"""

import asyncio
import logging
from typing import Optional

import chainlit as cl

from ..agent.graph import TeachingAgent
from ..teaching.methodology import TeachingMethodology
from .session_manager import SessionManager
from .file_handler import process_uploaded_files
from .history_manager import history_manager

logger = logging.getLogger(__name__)

# Global singletons
_agent: Optional[TeachingAgent] = None
_session_manager = SessionManager()

# Commands that open the conversation history view (instead of teaching)
HISTORY_COMMANDS = {"/history", "/历史", "查看历史", "历史记录", "查看历史记录"}

# Welcome message template
WELCOME_MESSAGE = """# 🎓 开源课程助教 Agent

你好！我是你的专属 AI 课程助教，采用 MIT/Stanford 教学风格。

### 我能做什么
- 📖 基于你上传的课程资料（notes, slides, textbooks），用中文为你**结构化讲解**知识点
- 🔤 保留所有英文专业术语和数学符号（$X$, $\\beta$, bias 等）
- 📝 按四段式结构讲解：**核心流程 → 示例展开 → 数学符号 → 总结提炼**

### 开始使用
1. **上传课程 PDF** — 在下方对话框中上传你的课程资料
2. **向我提问** — 比如 "Explain gradient descent" 或 "什么是 Fourier Transform？"
3. **深入学习** — 我会基于你的课程资料，逐层深入地为你讲解

> ⚠️ 注意：我只针对 notes/slides/textbooks 等知识性内容讲解。Labs、homeworks 和 projects 请你自己独立实践完成。"""


def _get_agent() -> TeachingAgent:
    """Get or create the teaching agent singleton.

    Safe without a lock: asyncio is single-threaded and this function
    has no await points, so it runs atomically from check to assignment.
    """
    global _agent
    if _agent is None:
        _agent = TeachingAgent()
    return _agent


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session.

    1. Create user session
    2. Send welcome message
    3. Prompt for initial course PDF upload
    4. Process uploaded files through ingestion pipeline
    """
    # Create user session
    user_id = cl.user_session.get("id", "default")
    session = _session_manager.get_or_create_session(user_id)

    # Store in Chainlit session
    cl.user_session.set("thread_id", session.thread_id)
    cl.user_session.set("user_session", session)

    # Create a persistent history record for this conversation
    history_manager.create_conversation(session.thread_id, user_id=user_id)

    # Initialize agent
    _get_agent()

    # Send welcome message
    welcome = WELCOME_MESSAGE + (
        "\n\n---\n\n"
        "💡 **小贴士**：输入 `/history` 可查看历史对话、上传过的文件和讲解记录。\n"
        "重复上传同一份 PDF 会被自动跳过，无需手动去重。"
    )
    await cl.Message(content=welcome).send()

    logger.info("Chat started: user=%s, thread=%s", user_id, session.thread_id)

    # Prompt for initial file upload
    try:
        files = await cl.AskFileMessage(
            content="📚 请上传你的课程资料（支持多选 PDF 文件）：",
            accept=["application/pdf"],
            max_files=10,
            timeout=3600,  # 1-hour timeout
        ).send()

        if files:
            # Process uploaded files
            status_msg = cl.Message(content="")
            await status_msg.send()

            success_count, total_chunks = await process_uploaded_files(
                files, status_msg, thread_id=session.thread_id
            )

            if success_count > 0:
                status_msg.content = (
                    f"✅ 知识库构建完成！\n\n"
                    f"- 成功处理: {success_count}/{len(files)} 个文件\n"
                    f"- 总共生成: {total_chunks} 个知识块（chunks）\n\n"
                    f"现在你可以基于这些资料向我提问了！"
                )
                await status_msg.update()
        else:
            await cl.Message(
                content="ℹ️ 你还没有上传课程资料。可以先向我提问，或随时上传 PDF 文件。"
            ).send()

    except Exception as e:
        logger.error("File upload flow failed: %s", e)
        await cl.Message(
            content=f"⚠️ 文件上传流程遇到问题: {str(e)}\n\n你可以先提问，或刷新页面重试。"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming student messages.

    Flow:
    1. Check for attached files → process if any
    2. Show thinking indicator
    3. Run LangGraph teaching pipeline
    4. Display structured response
    """
    # -- Step 0: Check for file attachments --
    thread_id = cl.user_session.get("thread_id", "default")
    elements = getattr(message, "elements", None)
    if elements:
        pdf_elements = [e for e in elements if getattr(e, "mime", "") == "application/pdf"
                       or getattr(e, "name", "").lower().endswith(".pdf")]
        if pdf_elements:
            status_msg = cl.Message(content="")
            await status_msg.send()
            success_count, total_chunks = await process_uploaded_files(
                pdf_elements, status_msg, thread_id=thread_id
            )
            if success_count > 0:
                status_msg.content = (
                    f"✅ 已处理 {success_count} 个文件，生成 {total_chunks} 个知识块。\n"
                    f"你可以基于新资料向我提问了！"
                )
                await status_msg.update()

    # -- Step 1: Handle text query --
    user_query = (message.content or "").strip()
    if not user_query:
        # User only uploaded files without a question
        if not elements:
            await cl.Message(content="请提出你的问题，我会尽力帮你解答！").send()
        return

    # -- Step 1b: History command — show past conversations/files/explanations --
    if user_query.lower() in HISTORY_COMMANDS:
        await _render_history_view()
        return

    logger.info("Received query [thread=%s]: %s", thread_id, user_query[:100])

    # -- Step 2: Show thinking indicator --
    thinking_msg = cl.Message(content="🤔 正在分析你的问题...")
    await thinking_msg.send()

    # -- Step 3: Run the teaching pipeline --
    agent = _get_agent()

    try:
        result = await agent.ateach(user_query, thread_id=thread_id)
    except Exception as e:
        logger.error("Teaching pipeline failed: %s", e)
        thinking_msg.content = f"❌ 处理你的问题时遇到错误: {str(e)}"
        await thinking_msg.update()
        return

    thinking_msg.content = "📝 正在整理讲解内容..."
    await thinking_msg.update()

    # -- Step 4: Assemble the response --
    methodology = TeachingMethodology()

    core = result.get("core_explanation", "")
    examples = result.get("examples", "")
    math_notation = result.get("math_notation", "")
    summary = result.get("section_summary", "")

    # Check intent for special handling
    intent = result.get("intent", "learn_new")
    if intent == "navigate":
        thinking_msg.content = core or "你好！有什么我可以帮助你的吗？"
        await thinking_msg.update()
        return

    if intent == "refuse":
        thinking_msg.content = core or (
            "⚠️ 我注意到你在询问 lab/homework/project 相关的内容。\n\n"
            "作为一名负责任的课程助教，我只针对 **notes、slides、textbooks** 等知识性内容进行讲解。\n\n"
            "Lab、homework 和 project 是为你设计的独立实践环节——**亲手解决问题才是真正学习的开始**。\n\n"
            "如果你对背后的概念原理感到困惑，请随时问我！比如：\n"
            "- 这个算法背后的 intuition 是什么？\n"
            "- 某个公式的含义和推导过程\n"
            "- 相关概念之间的区别与联系\n\n"
            "加油，你可以的！💪"
        )
        await thinking_msg.update()
        return

    # Assemble full four-part response
    full_response = methodology.assemble_full_response(
        core_explanation=core,
        examples=examples,
        math_notation=math_notation,
        section_summary=summary,
    )

    # -- Step 5: Display the response --
    await thinking_msg.remove()

    # If response is too long, split into multiple messages
    MAX_MSG_LENGTH = 4000
    if len(full_response) > MAX_MSG_LENGTH:
        sections = full_response.split("\n\n---\n\n")
        for i, section in enumerate(sections):
            if section.strip():
                await cl.Message(content=section.strip()).send()
                if i < len(sections) - 1:
                    await asyncio.sleep(0.3)
    else:
        await cl.Message(content=full_response).send()

    # Record the question
    user_session = cl.user_session.get("user_session")
    if user_session:
        _session_manager.record_question(
            user_session.user_id,
            result.get("target_topic", user_query),
        )

    # Persist this Q&A pair (with a short preview of the summary) to history
    try:
        summary_preview = (summary or core or "")[:200].replace("\n", " ")
        history_manager.record_qa(
            thread_id=thread_id,
            query=user_query,
            topic=result.get("target_topic", user_query),
            summary_preview=summary_preview,
        )
    except Exception as e:
        logger.warning("Failed to record Q&A in history: %s", e)

    # Show sources if available
    chunks = result.get("retrieved_chunks", [])
    if chunks:
        sources = set(c["source"] for c in chunks if c.get("source"))
        sources_text = "\n".join(f"- 📄 {s}" for s in sorted(sources))
        await cl.Message(
            content=f"**参考来源:**\n{sources_text}",
            author="系统",
        ).send()

    logger.info("Response sent [thread=%s]: %d chars", thread_id, len(full_response))


async def _render_history_view() -> None:
    """Render past conversations, their uploaded files, and Q&A summaries.

    Uses AskActionMessage with one Action per conversation so the student
    can click a button to view that conversation's full file/Q&A listing.
    """
    try:
        convs = history_manager.list_conversations()
    except Exception as e:
        logger.error("Failed to load history: %s", e)
        await cl.Message(content=f"⚠️ 读取历史记录失败: {e}").send()
        return

    if not convs:
        await cl.Message(
            content="📭 还没有历史记录。\n上传课程资料并提问后，这里会出现对话清单。"
        ).send()
        return

    overview_lines = ["## 📚 历史对话总览\n"]
    actions: list[cl.Action] = []
    MAX_SHOW = 8
    for idx, conv in enumerate(convs[:MAX_SHOW], start=1):
        created = conv.get("created_at", "?")
        files = conv.get("files", [])
        qa_list = conv.get("q_and_a", [])
        thread = conv.get("thread_id", "")
        overview_lines.append(
            f"**{idx}.** `{created}`  ·  📄 {len(files)} 个文件  ·  💬 {len(qa_list)} 次问答"
        )
        if files:
            overview_lines.append("   - 文件: " + ", ".join(
                f["name"] for f in files
            ))
        if qa_list:
            first_q = qa_list[0].get("query", "")[:40]
            overview_lines.append(f"   - 首问: {first_q}")
        overview_lines.append("")
        actions.append(cl.Action(
            name="view_conv",
            payload={"thread_id": thread, "idx": idx},
            label=f"查看第{idx}条",
            tooltip=f"展开 {created} 的文件与讲解记录",
            icon="",
        ))

    if len(convs) > MAX_SHOW:
        overview_lines.append(
            f"\n_（仅显示最近 {MAX_SHOW} 条，共 {len(convs)} 条历史记录。）_"
        )

    content = "\n".join(overview_lines)
    msg = cl.AskActionMessage(
        content=content + "\n点击下方按钮可展开对应文件的「文件 + 讲解摘要」明细：",
        actions=actions,
        timeout=600,
    )
    await msg.send()
    res = await msg.wait_for_answer()
    if res is None:
        return

    payload = res.get("payload", {})
    sel_thread = payload.get("thread_id")
    if not sel_thread:
        return

    conv = history_manager.get_conversation(sel_thread)
    if conv is None:
        await cl.Message(content="⚠️ 未找到该对话记录（可能已被清除）。").send()
        return

    await _render_conversation_detail(conv)


async def _render_conversation_detail(conv: dict) -> None:
    """Build a detailed markdown view for one conversation record."""
    lines = [
        f"## 🗂 对话明细  `{conv.get('created_at', '')}`\n",
        f"- Thread: `{conv.get('thread_id', '')}`",
        f"- 创建时间: {conv.get('created_at', '?')}",
        f"- 更新时间: {conv.get('updated_at', '?')}",
    ]

    files = conv.get("files", [])
    lines.append("\n### 📄 上传的文件\n")
    if not files:
        lines.append("- 无")
    else:
        lines.append("| 文件名 | Chunks | 内容指纹(前12位) |")
        lines.append("|---|---|---|")
        for f in files:
            h = (f.get("hash") or "")[:12]
            lines.append(f"| {f.get('name','')} | {f.get('chunks','')} | `{h}` |")

    qa_list = conv.get("q_and_a", [])
    lines.append("\n### 💬 提问与讲解摘要\n")
    if not qa_list:
        lines.append("- 该对话还没有提问记录。")
    else:
        for i, qa in enumerate(qa_list, start=1):
            lines.append(f"**Q{i}.** `{qa.get('time','')}` — 主题: {qa.get('topic','')}")
            lines.append(f"> 问：{qa.get('query','')}")
            preview = qa.get("summary_preview", "")
            if preview:
                lines.append(f"> 讲解摘要：{preview}…")
            lines.append("")

    lines.append(
        "\n---\n"
        "💡 提示：本对话的所有文件已索引进向量库。开启新对话时重新上传同一份 PDF "
        "会被自动跳过，无需手动去重。如需在原对话上继续追问，请回到该对话窗口。"
    )

    await cl.Message(content="\n".join(lines)).send()
