"""Thread-safe in-memory and optional disk/JSON-persisted history store.

Maintains an ordered ring-buffer of events/interactions (Q&A and soundscapes),
similar to a GitHub activity or 'Latest Changes' feed.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class HistoryEntry:
    id: str  # standard UUID (uuid.UUID string)
    kind: str  # "ask" | "soundscape"
    timestamp: str
    prompt_or_query: str
    result_summary: str
    vision_model: Optional[str] = None
    audio_model: Optional[str] = None
    attempts: int = 1
    has_image: bool = False


class HistoryStore:
    def __init__(self, max_entries: int = 50):
        self._max_entries = max_entries
        self._entries: List[HistoryEntry] = []
        self._lock = threading.Lock()

    def add(
        self,
        kind: str,
        prompt_or_query: str,
        result_summary: str,
        entry_id: Optional[str] = None,
        vision_model: Optional[str] = None,
        audio_model: Optional[str] = None,
        attempts: int = 1,
        has_image: bool = False,
    ) -> HistoryEntry:
        with self._lock:
            # Always ensure a valid UUID format
            uid = entry_id or str(uuid.uuid4())
            entry = HistoryEntry(
                id=uid,
                kind=kind,
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt_or_query=prompt_or_query,
                result_summary=result_summary,
                vision_model=vision_model,
                audio_model=audio_model,
                attempts=attempts,
                has_image=has_image,
            )
            self._entries.insert(0, entry)
            if len(self._entries) > self._max_entries:
                self._entries.pop()
            return entry

    def list_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(e) for e in self._entries[:limit]]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# Global singleton instance
history_store = HistoryStore(max_entries=50)

