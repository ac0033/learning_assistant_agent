"""Persistent conversation & file history manager.

Stores a JSON log of every teaching conversation so the student can:
  - Review past conversations, uploaded files, and the explanations given
  - Avoid re-uploading the same file (dedup by content hash)
  - Resume earlier threads without re-asking from scratch

Storage layout (under data/history/):
  conversations.json  — list of conversation records keyed by thread_id
  file_registry.json  — map content_hash -> {name, threads, chunks, first_uploaded}

All operations are atomic-ish (read-modify-write) and fail soft: any IO
error is logged and swallowed so the chat flow never crashes due to history.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()


def _history_dir() -> Path:
    d = settings.data_dir / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conv_path() -> Path:
    return _history_dir() / "conversations.json"


def _registry_path() -> Path:
    return _history_dir() / "file_registry.json"


def _utcnow_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compute_file_hash(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of file content. Used to detect re-uploads of the same file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s — using default", path, e)
        return default


def _write_json(path: Path, data: Any) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except OSError as e:
        logger.error("Failed to write %s: %s", path, e)


class HistoryManager:
    """Append/update-style persistent history backed by two JSON files.

    Thread-safe via a module-level lock; this matters because Chainlit
    may run sync helpers from a thread pool.
    """

    def __init__(self):
        self._lock = _LOCK

    # ------------------------------------------------------------------
    # Conversation records
    # ------------------------------------------------------------------
    def _load_conversations(self) -> dict[str, dict]:
        data = _read_json(_conv_path(), {})
        if isinstance(data, dict):
            return data
        logger.warning("conversations.json shape unexpected — reinitializing")
        return {}

    def _save_conversations(self, data: dict[str, dict]) -> None:
        _write_json(_conv_path(), data)

    def create_conversation(self, thread_id: str, user_id: str = "") -> dict:
        """Create a new conversation record if one does not already exist."""
        with self._lock:
            convs = self._load_conversations()
            if thread_id in convs:
                return convs[thread_id]
            record = {
                "thread_id": thread_id,
                "user_id": user_id,
                "created_at": _utcnow_iso(),
                "updated_at": _utcnow_iso(),
                "files": [],          # [{name, hash, chunks}]
                "q_and_a": [],        # [{time, query, topic, summary_preview}]
            }
            convs[thread_id] = record
            self._save_conversations(convs)
            logger.info("[History] Created conversation %s", thread_id)
            return record

    def get_conversation(self, thread_id: str) -> Optional[dict]:
        with self._lock:
            return self._load_conversations().get(thread_id)

    def get_thread_files(self, thread_id: str) -> list[str]:
        """Return the list of file names registered to a conversation thread.

        Used by the retrieval layer to restrict retrieval to the files the
        student actually uploaded in the current conversation, preventing
        cross-conversation contamination from globally shared ChromaDB.
        Returns an empty list for a fresh conversation with no uploads.
        """
        conv = self.get_conversation(thread_id)
        if not conv:
            return []
        return [f["name"] for f in conv.get("files", []) if f.get("name")]

    def register_file(self, thread_id: str, name: str, content_hash: str,
                      chunks: int) -> None:
        """Attach an uploaded file to a conversation record."""
        with self._lock:
            convs = self._load_conversations()
            conv = convs.get(thread_id)
            if conv is None:
                conv = self.create_conversation(thread_id)
                convs = self._load_conversations()
                conv = convs[thread_id]
            for existing in conv.get("files", []):
                if existing.get("hash") == content_hash:
                    existing["name"] = name
                    existing["chunks"] = chunks
                    break
            else:
                conv.setdefault("files", []).append({
                    "name": name, "hash": content_hash, "chunks": chunks,
                })
            conv["updated_at"] = _utcnow_iso()
            self._save_conversations(convs)
            logger.info("[History] Registered file %s (hash=%s) -> thread %s",
                        name, content_hash[:12], thread_id)

    def record_qa(self, thread_id: str, query: str, topic: str,
                  summary_preview: str = "",
                  full_response: str = "") -> None:
        """Append a Q&A entry to a conversation record.

        Args:
            full_response: the complete four-part teaching output as rendered
                to the UI. Stored so /history can re-display it verbatim
                without re-running the agent.
        """
        with self._lock:
            convs = self._load_conversations()
            conv = convs.get(thread_id)
            if conv is None:
                conv = self.create_conversation(thread_id)
                convs = self._load_conversations()
                conv = convs[thread_id]
            entry: dict[str, Any] = {
                "time": _utcnow_iso(),
                "query": query,
                "topic": topic,
                "summary_preview": summary_preview,
            }
            if full_response:
                entry["full_response"] = full_response
            conv.setdefault("q_and_a", []).append(entry)
            conv["updated_at"] = _utcnow_iso()
            self._save_conversations(convs)

    # ------------------------------------------------------------------
    # File registry (content hash -> dedup)
    # ------------------------------------------------------------------
    def _load_registry(self) -> dict[str, dict]:
        data = _read_json(_registry_path(), {})
        return data if isinstance(data, dict) else {}

    def is_file_known(self, content_hash: str) -> Optional[dict]:
        """Return the registry entry for a hash if the file was ever uploaded."""
        with self._lock:
            return self._load_registry().get(content_hash)

    def register_file_global(self, content_hash: str, name: str,
                             chunks: int, thread_id: str) -> None:
        with self._lock:
            reg = self._load_registry()
            entry = reg.get(content_hash, {})
            threads = set(entry.get("threads", []))
            threads.add(thread_id)
            reg[content_hash] = {
                "name": name,
                "chunks": chunks,
                "first_uploaded": entry.get("first_uploaded", _utcnow_iso()),
                "threads": sorted(threads),
            }
            _write_json(_registry_path(), reg)

    # ------------------------------------------------------------------
    # Listing for /history command
    # ------------------------------------------------------------------
    def list_conversations(self) -> list[dict]:
        with self._lock:
            convs = self._load_conversations()
        result = list(convs.values())
        result.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return result


# Global singleton
history_manager = HistoryManager()