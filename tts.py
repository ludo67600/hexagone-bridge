"""
TTS — Synthèse vocale via edge-tts (voix Microsoft françaises, gratuit).

Le pitch et le rate sont tirés au sort par le serveur FiveM à l'ouverture de
session et restent constants pendant toute la conversation : c'est ce qui donne
à chaque PNJ une voix reconnaissable du début à la fin.
"""

import asyncio
import os

import edge_tts

# Voix par défaut si celle demandée est inconnue.
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "fr-FR-HenriNeural")

# Pools de référence (le serveur FiveM fait le tirage, on valide juste ici).
VOICES_MALE = [
    "fr-FR-HenriNeural",
    "fr-FR-RemyMultilingualNeural",
    "fr-CA-ThierryNeural",
    "fr-BE-GerardNeural",
]
VOICES_FEMALE = [
    "fr-FR-DeniseNeural",
    "fr-FR-EloiseNeural",
    "fr-FR-VivienneMultilingualNeural",
    "fr-CA-SylvieNeural",
    "fr-CH-ArianeNeural",
]
ALL_VOICES = set(VOICES_MALE + VOICES_FEMALE)

# Voix encore listées par Microsoft mais qui ne renvoient plus de son : remplacées d'office
# (sinon le PNJ qui l'a tirée reste muet). Eloise : « No audio was received » le 28/10/2026.
VOIX_REMPLACEES = {
    "fr-FR-EloiseNeural": "fr-FR-DeniseNeural",
}

# Bornes de sécurité (cohérentes avec le tirage côté serveur FiveM)
PITCH_MIN, PITCH_MAX = -15, 15   # Hz
RATE_MIN, RATE_MAX = -10, 15     # %

TIMEOUT = float(os.getenv("TTS_TIMEOUT", "12"))


def _clamp(value: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    return max(lo, min(hi, v))


async def synthesize(text: str, voice: str | None = None, pitch: int = 0, rate: int = 0) -> bytes:
    """
    Synthétise du texte en MP3.

    :param voice: nom de voix edge-tts (ex. 'fr-FR-HenriNeural')
    :param pitch: décalage en Hz (-15 à +15)
    :param rate: variation de débit en % (-10 à +15)
    :return: octets MP3 (vides en cas d'échec)
    """
    text = (text or "").strip()
    if not text:
        return b""

    if not voice or voice not in ALL_VOICES:
        voice = DEFAULT_VOICE
    voice = VOIX_REMPLACEES.get(voice, voice)

    pitch = _clamp(pitch, PITCH_MIN, PITCH_MAX)
    rate = _clamp(rate, RATE_MIN, RATE_MAX)

    audio = await _synthese(text, voice, pitch, rate)
    if not audio:
        # Voix en panne chez Microsoft : nouvel essai avec la voix principale du même genre
        secours = VOICES_FEMALE[0] if voice in VOICES_FEMALE else DEFAULT_VOICE
        if secours != voice:
            print(f"[tts] {voice} muette, nouvel essai avec {secours}")
            audio = await _synthese(text, secours, pitch, rate)
    return audio


async def _synthese(text: str, voice: str, pitch: int, rate: int) -> bytes:
    # edge-tts attend des chaînes signées : "+5Hz", "-3%"
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=f"{rate:+d}%",
        pitch=f"{pitch:+d}Hz",
    )

    audio = bytearray()

    async def _collect():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])

    try:
        await asyncio.wait_for(_collect(), timeout=TIMEOUT)
    except asyncio.TimeoutError:
        print("[tts] Timeout de synthèse")
        return b""
    except Exception as exc:
        print(f"[tts] Erreur edge-tts : {exc}")
        return b""

    return bytes(audio)
