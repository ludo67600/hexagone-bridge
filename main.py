"""
Bridge IA pour PNJ FiveM — FastAPI.

Pipeline : audio (webm/opus) → STT Groq → LLM Groq → TTS edge-tts → MP3 base64.

Le serveur FiveM est le SEUL client de ce service : il envoie l'audio du joueur
avec le contexte de la session, et reçoit la réplique audio du PNJ.
La clé Groq ne quitte jamais cette machine.

Démarrage :
    uvicorn main:app --host 0.0.0.0 --port 8080
"""

import base64
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

import cache
import llm
import stt
import tts

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "")          # partagé avec le serveur FiveM
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(2 * 1024 * 1024)))  # 2 Mo

# Défense en profondeur : le serveur FiveM limite déjà, on se protège aussi.
RL_MIN_INTERVAL = float(os.getenv("RL_MIN_INTERVAL", "3"))   # 1 requête / 3 s / joueur
RL_HOURLY_MAX = int(os.getenv("RL_HOURLY_MAX", "20"))        # 20 / heure / joueur

_last_call: dict[str, float] = {}
_hourly: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await cache.init()
    print("[bridge] Cache prêt —", cache.stats_sync())
    if not BRIDGE_TOKEN:
        print("[bridge] ⚠ BRIDGE_TOKEN vide : l'authentification est DÉSACTIVÉE (dev uniquement)")
    yield


app = FastAPI(title="Hexagone AI NPC Bridge", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Modèles de requête
# --------------------------------------------------------------------------

def _empty_dict_as_list(v):
    """Lua ne distingue pas table vide et objet vide : json.encode({}) donne "{}".

    Sans cette tolérance, toute liste vide envoyée par FiveM (history au premier
    message, items d'un joueur sans inventaire) déclenche une erreur 422.
    """
    if isinstance(v, dict) and not v:
        return []
    return v


class NpcModel(BaseModel):
    id: str = "npc"
    name: str = "Habitant"
    job: str = ""
    personality: str = ""
    knows: str = ""
    ignores: str = ""
    tone: str = ""
    allowed_actions: list[str] = Field(default_factory=list)

    _fix_actions = field_validator("allowed_actions", mode="before")(_empty_dict_as_list)


class PlayerModel(BaseModel):
    id: str = "0"                 # identifiant serveur, sert au rate-limit
    name: str = ""
    job: str = ""
    grade: str = ""
    money: int | None = None
    items: list[str] = Field(default_factory=list)
    appearance: str = ""          # apparence perçue par le PNJ (genre, arme, masque...)

    _fix_items = field_validator("items", mode="before")(_empty_dict_as_list)


class WorldModel(BaseModel):
    time: str = ""
    weather: str = ""
    location: str = ""            # rue / quartier où se déroule la scène


class VoiceModel(BaseModel):
    name: str = ""
    pitch: int = 0
    rate: int = 0


class TalkRequest(BaseModel):
    audio_b64: str = ""                       # segment audio du joueur (webm/opus)
    text: str = ""                            # alternative : texte direct (tests / PNJ écrits)
    npc: NpcModel = Field(default_factory=NpcModel)
    player: PlayerModel = Field(default_factory=PlayerModel)
    world: WorldModel = Field(default_factory=WorldModel)
    voice: VoiceModel = Field(default_factory=VoiceModel)
    history: list[dict] = Field(default_factory=list)

    _fix_history = field_validator("history", mode="before")(_empty_dict_as_list)


# --------------------------------------------------------------------------
# Sécurité / limites
# --------------------------------------------------------------------------

def _check_auth(authorization: str | None) -> None:
    if not BRIDGE_TOKEN:
        return  # mode dev
    expected = f"Bearer {BRIDGE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Token invalide")


def _check_rate_limit(player_id: str) -> None:
    now = time.time()

    last = _last_call.get(player_id, 0)
    if now - last < RL_MIN_INTERVAL:
        raise HTTPException(status_code=429, detail="Trop rapide, patientez un instant")

    window = _hourly[player_id]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RL_HOURLY_MAX:
        raise HTTPException(status_code=429, detail="Quota horaire atteint")

    _last_call[player_id] = now
    window.append(now)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "ok": True,
        "stt_model": stt.MODEL,
        "llm_model": llm.MODEL,
        "cache": cache.stats_sync(),
        "auth": bool(BRIDGE_TOKEN),
    }


@app.post("/talk")
async def talk(req: TalkRequest, authorization: str | None = Header(default=None)):
    """Traite un segment de parole et renvoie la réponse audio du PNJ."""
    _check_auth(authorization)
    _check_rate_limit(req.player.id or "0")

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    # ---------------- 1. STT ----------------
    transcription = (req.text or "").strip()
    if not transcription:
        if not req.audio_b64:
            raise HTTPException(status_code=400, detail="audio_b64 ou text requis")
        try:
            audio_in = base64.b64decode(req.audio_b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="audio_b64 invalide")

        if len(audio_in) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Segment audio trop volumineux")

        t = time.perf_counter()
        try:
            transcription = await stt.transcribe(audio_in)
        except Exception as exc:
            print(f"[bridge] STT échec : {exc}")
            raise HTTPException(status_code=502, detail="Transcription indisponible")
        timings["stt"] = round(time.perf_counter() - t, 3)

    # Rien d'audible : on ne réveille ni le LLM ni la TTS.
    if not transcription:
        return {
            "ok": True,
            "empty": True,
            "transcription": "",
            "speech": "",
            "action": {"type": "none"},
            "audio_b64": "",
            "timings": {"total": round(time.perf_counter() - t0, 3)},
        }

    # ---------------- 2. LLM (avec cache) ----------------
    allowed = req.npc.allowed_actions or []
    llm_key = cache.make_key("llm", req.npc.id, transcription)

    cached_llm = await cache.get_llm(llm_key) if not req.history else None
    # NB : on ne sert le cache que hors contexte conversationnel, sinon le PNJ
    # répondrait à côté (une même phrase n'a pas le même sens au 4e échange).

    if cached_llm:
        speech, action = cached_llm["speech"], cached_llm["action"]
        timings["llm"] = 0.0
        llm_cached = True
    else:
        t = time.perf_counter()
        result = await llm.generate(
            npc=req.npc.model_dump(),
            player=req.player.model_dump(),
            world=req.world.model_dump(),
            history=req.history,
            user_text=transcription,
            allowed=allowed,
        )
        timings["llm"] = round(time.perf_counter() - t, 3)
        speech, action = result["speech"], result["action"]
        llm_cached = False
        if not req.history:
            await cache.set_llm(llm_key, speech, action)

    # ---------------- 3. TTS (avec cache) ----------------
    tts_key = cache.make_key("tts", speech, req.voice.name, req.voice.pitch, req.voice.rate)
    audio_out = await cache.get_tts(tts_key)
    tts_cached = audio_out is not None

    if not tts_cached:
        t = time.perf_counter()
        audio_out = await tts.synthesize(
            speech, voice=req.voice.name, pitch=req.voice.pitch, rate=req.voice.rate
        )
        timings["tts"] = round(time.perf_counter() - t, 3)
        if audio_out:
            await cache.set_tts(tts_key, audio_out)
    else:
        timings["tts"] = 0.0

    timings["total"] = round(time.perf_counter() - t0, 3)

    return {
        "ok": True,
        "transcription": transcription,
        "speech": speech,
        "action": action,
        "audio_b64": base64.b64encode(audio_out).decode("ascii") if audio_out else "",
        "cached": {"llm": llm_cached, "tts": tts_cached},
        "timings": timings,
    }


@app.post("/purge")
async def purge_cache(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    await cache.purge()
    return {"ok": True, "cache": cache.stats_sync()}
