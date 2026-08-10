"""
Cache SQLite — TTS uniquement.

  • TTS : clé = hash(texte + voix + pitch + rate)
        L'audio dépend de la voix tirée pour la session : on ne peut pas
        réutiliser le MP3 d'une session à l'autre.
"""

import asyncio
import hashlib
import os
import sqlite3
import threading
import time

DB_PATH = os.getenv("CACHE_DB", "cache.sqlite3")

TTS_TTL = int(os.getenv("CACHE_TTS_TTL", str(7 * 24 * 3600)))     # 7 jours

# Garde-fous de taille (l'audio pèse lourd)
MAX_TTS_ROWS = int(os.getenv("CACHE_MAX_TTS_ROWS", "1500"))

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def _init_sync() -> None:
    conn = _connect()
    with _lock:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tts_cache (
                key        TEXT PRIMARY KEY,
                audio      BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)
        conn.commit()


async def init() -> None:
    """À appeler au démarrage de l'application."""
    await asyncio.to_thread(_init_sync)
    await asyncio.to_thread(_purge_sync)


def make_key(*parts) -> str:
    """Clé de cache stable à partir de n'importe quels éléments."""
    raw = "\x1f".join(str(p).strip().lower() for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------

def _get_tts_sync(key: str):
    conn = _connect()
    with _lock:
        row = conn.execute(
            "SELECT audio, created_at FROM tts_cache WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return None
    audio, created = row
    if time.time() - created > TTS_TTL:
        return None
    return bytes(audio)


def _set_tts_sync(key: str, audio: bytes) -> None:
    if not audio:
        return
    conn = _connect()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO tts_cache (key, audio, created_at) VALUES (?, ?, ?)",
            (key, audio, int(time.time())),
        )
        conn.commit()


async def get_tts(key: str):
    return await asyncio.to_thread(_get_tts_sync, key)


async def set_tts(key: str, audio: bytes) -> None:
    await asyncio.to_thread(_set_tts_sync, key, audio)


# --------------------------------------------------------------------------
# Purge
# --------------------------------------------------------------------------

def _purge_sync() -> None:
    """Supprime les entrées expirées puis borne la taille des tables."""
    conn = _connect()
    now = int(time.time())
    with _lock:
        conn.execute("DELETE FROM tts_cache WHERE ? - created_at > ?", (now, TTS_TTL))

        # Ne garder que les N plus récentes
        conn.execute("""
            DELETE FROM tts_cache WHERE key NOT IN (
                SELECT key FROM tts_cache ORDER BY created_at DESC LIMIT ?
            )
        """, (MAX_TTS_ROWS,))
        conn.commit()


async def purge() -> None:
    await asyncio.to_thread(_purge_sync)


def stats_sync() -> dict:
    conn = _connect()
    with _lock:
        tts = conn.execute("SELECT COUNT(*) FROM tts_cache").fetchone()[0]
    return {"tts_entries": tts}
