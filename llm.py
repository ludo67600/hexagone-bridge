"""
LLM — Génération de la réponse du PNJ via Groq (llama-3.3-70b-versatile).

Le modèle renvoie du JSON structuré :
    {"speech": "...", "action": {"type": "follow", "target": "player"}}

Rien de ce qui sort d'ici n'est digne de confiance : la validation locale
(whitelist d'actions, filtre hors-RP, troncature) est la première barrière,
et le serveur FiveM revalide ensuite tout ce qui a un effet économique.
"""

import json
import os
import re

from groq import AsyncGroq

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Longueur max de la réplique : les PNJ répondent court (latence + crédibilité).
MAX_SPEECH_CHARS = int(os.getenv("LLM_MAX_SPEECH_CHARS", "280"))

# Types d'actions connus du système (le config FiveM restreint encore par PNJ).
ACTION_TYPES = {
    "none",
    "follow",
    "stop_follow",
    "goto",
    "enter_vehicle",
    "flee",
    "give_item",
    "give_money",
    "open_shop",
    "hands_up",
    "end_conversation",
}

# Description courte injectée dans le prompt pour que le modèle sache quand agir.
ACTION_HELP = {
    "follow": "suivre le joueur (il te demande de l'accompagner)",
    "stop_follow": "arrêter de suivre le joueur",
    "goto": "te rendre à un endroit précis",
    "enter_vehicle": "monter dans le véhicule du joueur",
    "flee": "fuir (si tu as peur ou qu'on te menace)",
    "give_item": "remettre un objet au joueur",
    "give_money": "donner quelques billets au joueur (précise le champ \"amount\", petit montant)",
    "open_shop": "ouvrir ton commerce / ton menu de vente",
    "hands_up": "lever les mains (si on te braque)",
    "end_conversation": "mettre fin à la conversation (au revoir, tu t'en vas)",
}

# Marqueurs indiquant que le modèle est sorti du rôle.
_OUT_OF_RP = re.compile(
    r"\b(intelligence artificielle|mod[èe]le de langage|langage model|"
    r"je suis une ia\b|en tant qu'?ia\b|chatgpt|openai|assistant virtuel|"
    r"je ne suis qu'un programme)\b",
    re.IGNORECASE,
)

FALLBACK_SPEECH = "Hmm... désolé, je n'ai pas bien saisi. Vous pouvez répéter ?"

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY manquante (variable d'environnement)")
        _client = AsyncGroq(api_key=key)
    return _client


def build_system_prompt(npc: dict, player: dict, world: dict, allowed: list[str]) -> str:
    """Construit la fiche de personnage + le contexte + les règles de sortie."""

    # --- Fiche du PNJ ---
    lines = [
        f"Tu incarnes {npc.get('name', 'un habitant')}, {npc.get('job', 'habitant de Los Santos')}.",
        f"Personnalité : {npc.get('personality', 'ordinaire, neutre')}.",
        f"Ton : {npc.get('tone', 'naturel, familier')}.",
    ]
    if npc.get("knows"):
        lines.append(f"Ce que tu sais : {npc['knows']}.")
    if npc.get("ignores"):
        lines.append(f"Ce que tu ignores totalement : {npc['ignores']}. Si on t'en parle, dis que tu n'en sais rien.")

    # --- Contexte joueur / monde ---
    ctx = []
    if player.get("name"):
        ctx.append(f"il se présente comme {player['name']}")
    if player.get("job"):
        grade = f" ({player['grade']})" if player.get("grade") else ""
        ctx.append(f"métier : {player['job']}{grade}")
    if player.get("money") is not None:
        ctx.append(f"argent visible : {player['money']}$")
    if player.get("items"):
        ctx.append("objets visibles : " + ", ".join(map(str, player["items"])))
    if player.get("appearance"):
        ctx.append(f"tu la vois ainsi : {player['appearance']}")
    if ctx:
        lines.append("La personne en face de toi : " + " ; ".join(ctx) + ".")

    wctx = []
    if world.get("time"):
        wctx.append(f"il est {world['time']}")
    if world.get("weather"):
        wctx.append(f"météo : {world['weather']}")
    if world.get("location"):
        wctx.append(f"vous vous trouvez à {world['location']}")
    if wctx:
        lines.append("Contexte : " + ", ".join(wctx) + ".")
        lines.append("Tu peux évoquer ce décor (le lieu, l'heure, la météo, l'allure du visiteur) si c'est naturel, mais sans réciter ces informations.")

    # --- Règles de jeu ---
    lines += [
        "",
        "RÈGLES ABSOLUES :",
        "- Réponds UNIQUEMENT en français, à l'oral, en 1 à 2 phrases courtes.",
        "- Ta réponse sera lue par une synthèse vocale : pas de listes, pas de markdown,",
        "  pas d'emoji, pas de didascalies entre astérisques. Uniquement des mots prononçables.",
        "- Reste TOUJOURS dans la peau du personnage. Ne dis jamais que tu es une IA,",
        "  un modèle, un assistant ou un programme, même si on te le demande.",
        "- Tu peux refuser en RP (méfiance, prix trop bas, mauvais interlocuteur...).",
        "- Si on te demande quelque chose que ton personnage ne ferait pas, refuse en restant crédible.",
    ]

    # --- Actions autorisées ---
    usable = [a for a in allowed if a in ACTION_HELP]
    if usable:
        lines.append("")
        lines.append("ACTIONS que tu peux déclencher (uniquement si c'est justifié) :")
        for a in usable:
            lines.append(f"- \"{a}\" : {ACTION_HELP[a]}")
        lines.append("N'utilise une action que si la demande du joueur le justifie vraiment.")
    else:
        lines.append("")
        lines.append("Tu ne peux déclencher AUCUNE action : utilise toujours \"none\".")

    # --- Format de sortie ---
    lines += [
        "",
        "FORMAT DE RÉPONSE — réponds STRICTEMENT avec cet objet JSON, rien d'autre :",
        '{"speech": "ta réplique parlée", "action": {"type": "none"}}',
        'Pour une action : {"speech": "...", "action": {"type": "follow", "target": "player"}}',
        'Pour donner un objet : {"speech": "...", "action": {"type": "give_item", "item": "bandage", "count": 1}}',
        'Pour donner de l\'argent : {"speech": "...", "action": {"type": "give_money", "amount": 20}}',
    ]

    return "\n".join(lines)


def _parse_json(raw: str) -> dict | None:
    """Parse la sortie du modèle en tolérant les blocs de code éventuels."""
    if not raw:
        return None
    txt = raw.strip()

    # Retire un éventuel ```json ... ```
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Dernier recours : isoler le premier objet JSON de la chaîne
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _sanitize(data: dict | None, allowed: list[str]) -> dict:
    """
    Valide et nettoie la sortie du modèle.
    Toute action non autorisée est écrasée par "none" (jamais rejetée silencieusement
    côté joueur : le PNJ parle quand même, il n'agit simplement pas).
    """
    speech = ""
    action = {"type": "none"}

    if isinstance(data, dict):
        raw_speech = data.get("speech")
        if isinstance(raw_speech, str):
            speech = raw_speech.strip()

        raw_action = data.get("action")
        if isinstance(raw_action, dict):
            atype = str(raw_action.get("type", "none")).strip().lower()
            if atype in ACTION_TYPES and atype in set(allowed) | {"none"}:
                action = {k: v for k, v in raw_action.items()}
                action["type"] = atype
            # sinon : action refusée → on garde "none"

    # Nettoyage du texte : la TTS ne doit pas lire des symboles.
    speech = re.sub(r"[*_`#]+", "", speech)
    speech = re.sub(r"\s+", " ", speech).strip()

    # Filtre hors-RP
    if not speech or _OUT_OF_RP.search(speech):
        speech = FALLBACK_SPEECH
        action = {"type": "none"}

    if len(speech) > MAX_SPEECH_CHARS:
        speech = speech[:MAX_SPEECH_CHARS].rsplit(" ", 1)[0] + "..."

    return {"speech": speech, "action": action}


async def generate(
    npc: dict,
    player: dict,
    world: dict,
    history: list[dict],
    user_text: str,
    allowed: list[str],
) -> dict:
    """
    Génère la réplique du PNJ.

    :param history: liste de {"role": "user"|"assistant", "content": "..."} (5 derniers échanges)
    :return: {"speech": str, "action": dict}
    """
    client = _get_client()

    messages = [{"role": "system", "content": build_system_prompt(npc, player, world, allowed)}]

    # Historique récent (on borne à 10 messages = 5 échanges)
    for h in (history or [])[-10:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})

    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
    except Exception as exc:  # panne API, quota, timeout...
        print(f"[llm] Erreur Groq : {exc}")
        return {"speech": FALLBACK_SPEECH, "action": {"type": "none"}}

    return _sanitize(_parse_json(raw), allowed)
