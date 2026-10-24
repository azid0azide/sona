"""Supabase client helper and database sync for Gemma & Soundscape interactions.

Every entry is saved with a canonical UUID (UUID v4) as primary key.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import uuid

log = logging.getLogger(__name__)

_supabase_client = None


def get_supabase_client() -> Optional[Any]:
    """Return the Supabase client if configured, otherwise None.

    Imported lazily so tests and offline runs never fail if supabase package
    is missing or if credentials are not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        log.warning("SUPABASE_URL or SUPABASE_KEY not set. Skipping Supabase sync.")
        return None

    try:
        from supabase import Client, create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        log.warning("Failed to initialize Supabase client: %s", exc)
        return None


def store_interaction(
    kind: str,
    prompt_or_query: str,
    result_summary: str,
    entry_uuid: Optional[str] = None,
    vision_model: Optional[str] = None,
    audio_model: Optional[str] = None,
    attempts: int = 1,
    has_image: bool = False,
    table_name: str = "interactions",
) -> Optional[Dict[str, Any]]:
    """Store an interaction record in Supabase with a standard UUID.

    If entry_uuid is not provided, a random UUID v4 is generated.
    """
    client = get_supabase_client()
    if not client:
        return None

    # Always ensure a valid UUID format
    record_id = entry_uuid or str(uuid.uuid4())

    payload = {
        "id": record_id,
        "kind": kind,
        "prompt_or_query": prompt_or_query,
        "result_summary": result_summary,
        "vision_model": vision_model,
        "audio_model": audio_model,
        "attempts": attempts,
        "has_image": has_image,
    }

    try:
        response = client.table(table_name).insert(payload).execute()
        log.info("Saved interaction to Supabase table '%s' [UUID: %s]", table_name, record_id)
        return response.data[0] if getattr(response, "data", None) else payload
    except Exception as exc:
        log.warning("Supabase insert failed for UUID %s (check table schema): %s", record_id, exc)
        return None


def fetch_interactions(limit: int = 20, table_name: str = "interactions") -> Optional[List[Dict[str, Any]]]:
    """Fetch recent interactions from Supabase."""
    client = get_supabase_client()
    if not client:
        return None

    try:
        response = client.table(table_name).select("*").order("created_at", desc=True).limit(limit).execute()
        return response.data if getattr(response, "data", None) else []
    except Exception as exc:
        log.warning("Supabase fetch failed: %s", exc)
        return None

