"""
STT — Transcription vocale via Groq (whisper-large-v3-turbo).

Le format webm/opus produit par MediaRecorder (NUI) est accepté tel quel par
l'API Groq : aucune conversion ffmpeg n'est nécessaire.
"""

import os
from groq import AsyncGroq

MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Client Groq partagé (créé à la première utilisation)."""
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY manquante (variable d'environnement)")
        _client = AsyncGroq(api_key=key)
    return _client


async def transcribe(audio: bytes, filename: str = "audio.webm", language: str = "fr") -> str:
    """
    Transcrit un segment audio en texte.

    :param audio: octets bruts (webm/opus, mp3, wav, ogg... — formats acceptés par Groq)
    :param filename: nom de fichier (l'extension aide l'API à détecter le format)
    :param language: code langue ISO ('fr' pour forcer le français et gagner en précision)
    :return: transcription nettoyée (chaîne vide si rien n'a été compris)
    """
    if not audio:
        return ""

    client = _get_client()
    resp = await client.audio.transcriptions.create(
        file=(filename, audio),
        model=MODEL,
        language=language,
        response_format="text",
        temperature=0.0,          # déterministe : on ne veut pas d'inventions
    )

    # response_format="text" renvoie directement une chaîne, mais on reste tolérant
    text = resp if isinstance(resp, str) else getattr(resp, "text", "") or ""
    return text.strip()
