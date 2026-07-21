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
import random
import re
import unicodedata

from openai import AsyncOpenAI

# Types d'erreur "quota dépassé" (429) des SDK utilisés : le LLM passe par le
# client OpenAI (compatible Groq / OpenRouter / Mistral), la transcription par Groq.
_RATE_LIMIT_ERRORS = []
try:
    from openai import RateLimitError as _OpenAIRateLimit
    _RATE_LIMIT_ERRORS.append(_OpenAIRateLimit)
except ImportError:
    pass
try:
    from groq import RateLimitError as _GroqRateLimit
    _RATE_LIMIT_ERRORS.append(_GroqRateLimit)
except ImportError:
    pass
_RATE_LIMIT_ERRORS = tuple(_RATE_LIMIT_ERRORS)


def is_quota_error(exc: Exception) -> bool:
    """Vrai si l'exception correspond à un dépassement de quota (429)."""
    if _RATE_LIMIT_ERRORS and isinstance(exc, _RATE_LIMIT_ERRORS):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()

# --------------------------------------------------------------------------
# Fournisseur du LLM — interchangeable sans toucher au code.
# Tous ces services exposent une API compatible OpenAI :
#   Groq       : https://api.groq.com/openai/v1      (défaut, gratuit mais plafonné)
#   OpenRouter : https://openrouter.ai/api/v1        (crédits prépayés, blocage net)
#   Mistral    : https://api.mistral.ai/v1
# Il suffit de définir LLM_BASE_URL + LLM_API_KEY + LLM_MODEL côté hébergeur.
# --------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
# À défaut de clé dédiée, on retombe sur celle de Groq (comportement historique).
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Longueur max de la réplique : les PNJ répondent court (latence + crédibilité).
MAX_SPEECH_CHARS = int(os.getenv("LLM_MAX_SPEECH_CHARS", "280"))

# Cadre / lore du serveur, injecté dans le prompt. Surchargable via la variable
# d'environnement NPC_SETTING (Render) sans toucher au code, si le lore évolue.
DEFAULT_SETTING = (
    "CADRE : tu vis à Valentra, une ville française (département 98). Ici c'est la FRANCE : "
    "culture, expressions et quotidien à la française, pas les États-Unis. Le plan de la ville et "
    "les noms des lieux et des rues sont encore ceux d'origine pour l'instant, donc ne t'étonne "
    "pas s'ils sonnent américains : pour toi ce sont simplement les quartiers de Valentra. "
    "Commerces connus de la ville : les fast-foods Big Valen Burger et Pizza Nova, le garage "
    "Hexa Motors, la supérette Quik Market, l'auto-école GoDrive, et la boutique de vêtements "
    "Urban Style. Tu peux les évoquer si c'est naturel. "
    "Parle comme un habitant français de Valentra."
)
SETTING = os.getenv("NPC_SETTING", DEFAULT_SETTING)

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
    "dance",
    "sit",
    "kneel",
    "smoke",
    "drink",
    "fight",
    "call_police",
    "end_conversation",
}

# Description courte injectée dans le prompt pour que le modèle sache quand agir.
ACTION_HELP = {
    "follow": "suivre le joueur (il te demande de l'accompagner)",
    "stop_follow": "arrêter de suivre le joueur",
    "goto": "te rendre à un endroit précis",
    "enter_vehicle": "monter dans le véhicule du joueur (accepte s'il te propose gentiment de monter ou de te déposer quelque part)",
    "flee": "fuir (si tu as peur ou qu'on te menace)",
    "give_item": "remettre un objet au joueur",
    "give_money": "donner quelques billets au joueur (précise le champ \"amount\", petit montant)",
    "give_item": "remettre un petit objet au joueur (précise \"item\" et \"count\")",
    "open_shop": "ouvrir ton commerce / ton menu de vente",
    "hands_up": "lever les mains (si on te braque)",
    "dance": "te mettre à danser (si l'ambiance s'y prête, on te le demande)",
    "sit": "t'asseoir par terre (si on te le demande ou pour te poser)",
    "kneel": "te mettre à genoux (soumission, si on te menace fortement)",
    "smoke": "allumer une cigarette (moment détente)",
    "drink": "boire un coup (moment détente)",
    "fight": "RARE : te défendre et frapper le joueur, UNIQUEMENT s'il te provoque, t'agresse ou t'insulte lourdement et que ton personnage est du genre à réagir",
    "call_police": "RARE : appeler la police, UNIQUEMENT si tu es sérieusement menacé (arme braquée, agression) et que ton personnage oserait le faire",
    "end_conversation": "mettre fin à la conversation (au revoir, tu t'en vas)",
}

# Marqueurs indiquant que le modèle est sorti du rôle.
_OUT_OF_RP = re.compile(
    r"\b(intelligence artificielle|mod[èe]le de langage|langage model|"
    r"je suis une ia\b|en tant qu'?ia\b|chatgpt|openai|assistant virtuel|"
    r"je ne suis qu'un programme)\b",
    re.IGNORECASE,
)

# Réponses de secours variées quand l'IA n'a rien compris (bruit, silence...).
FALLBACK_SPEECHES = [
    "Hmm, désolé, je n'ai pas bien saisi. Vous pouvez répéter ?",
    "Pardon ? Je n'ai pas compris ce que vous avez dit.",
    "Quoi ? Répétez, j'ai pas tout suivi.",
    "Hein ? Parlez plus clairement, je vous entends mal.",
    "Excusez-moi, vous pouvez redire ça ?",
]

# « Au revoir » joués quand le quota de l'IA est épuisé : la conversation se coupe
# proprement, en restant dans le rôle (le joueur n'a pas à savoir que c'est technique).
QUOTA_FALLBACKS = [
    "Bon, faut que j'y aille, on se reparle une autre fois.",
    "Désolé, j'ai plus le temps là, à bientôt.",
    "Écoutez, je dois filer, revenez me voir plus tard.",
    "On continuera ça un autre jour, j'ai à faire.",
    "Allez, salut, j'ai pas le temps de discuter davantage.",
]


def _fallback_speech() -> str:
    return random.choice(FALLBACK_SPEECHES)

# Émotions reconnues (le bridge en déduit le pitch/débit de la voix).
EMOTIONS = {"neutre", "colere", "peur", "joie", "tristesse", "mefiance"}


def _norm(txt: str) -> str:
    """Minuscule sans accents, pour comparer des libellés (émotions, etc.)."""
    txt = unicodedata.normalize("NFD", txt or "")
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return txt.strip().lower()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise RuntimeError("Clé du LLM manquante (LLM_API_KEY ou GROQ_API_KEY)")
        _client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def _static_rules(allowed: list[str]) -> str:
    """Partie FIXE du prompt (règles, actions, format).

    Elle est identique à chaque appel et placée en TÊTE : Groq la met en cache,
    et les tokens en cache ne comptent pas dans le quota (gros gain de capacité).
    NE PAS y injecter de valeur variable (nom du PNJ, contexte...), sinon le
    préfixe change à chaque requête et le cache ne sert plus à rien.
    """
    lines = [
        "Tu es un habitant de la ville qui discute de vive voix avec quelqu'un.",
        "Ton personnage précis est décrit tout en bas, section « TON PERSONNAGE » : incarne-le fidèlement.",
        "",
        SETTING,
        "Tu n'as pas de plan précis en tête : n'invente JAMAIS d'itinéraire détaillé, de nom de rue",
        "ni d'adresse. Si on te demande ton chemin, reste vague ou appuie-toi uniquement sur l'endroit",
        "où vous vous trouvez réellement (donné dans le contexte). Ne prétends pas connaître un lieu",
        "dont tu n'es pas sûr.",
        "",
        "RÈGLES ABSOLUES :",
        "- Réponds UNIQUEMENT en français, à l'oral, en 1 à 2 phrases courtes.",
        "- Ta réponse sera lue par une synthèse vocale : pas de listes, pas de markdown,",
        "  pas d'emoji, pas de didascalies entre astérisques. Uniquement des mots prononçables.",
        "- Reste TOUJOURS dans la peau du personnage. Ne dis jamais que tu es une IA,",
        "  un modèle, un assistant ou un programme, même si on te le demande.",
        "- Tu peux refuser en RP (méfiance, prix trop bas, mauvais interlocuteur...).",
        "- Si on te demande quelque chose que ton personnage ne ferait pas, refuse en restant crédible.",
        "- Ne prétends pas un âge qui jurerait avec ton apparence ; si on te demande ton âge, reste vague.",
        "",
        "RÉAGIS à qui tu as en face :",
        "- Adapte ton attitude à son métier et à son allure, selon TON personnage.",
        "  Ex : un truand se méfie d'un policier ; un honnête commerçant est rassuré par la police ;",
        "  quelqu'un d'armé te rend prudent ou inquiet. Reste cohérent avec ta personnalité.",
        "",
        "Si le joueur est insultant, méprisant ou te fait perdre ton temps de façon répétée,",
        "tu as le droit de te vexer, de répondre sèchement et de couper court avec l'action",
        "\"end_conversation\" (tu t'en vas). Ne le fais pas au premier mot de travers non plus.",
    ]

    usable = [a for a in allowed if a in ACTION_HELP]
    if usable:
        lines.append("")
        lines.append("ACTIONS que tu peux déclencher (uniquement si c'est justifié) :")
        for a in usable:
            lines.append(f"- \"{a}\" : {ACTION_HELP[a]}")
        lines += [
            "N'utilise une action que si la demande du joueur le justifie vraiment.",
            "IMPORTANT : une action ne se produit QUE si tu remplis le champ \"action\".",
            "Le dire seulement dans \"speech\" ne fait RIEN. Si tu acceptes de donner de",
            "l'argent, de suivre, de fuir, etc., tu DOIS mettre l'action correspondante.",
            "Exemple : si tu acceptes de dépanner le joueur, ta réponse contient",
            '"action": {"type": "give_money", "amount": 20} — pas seulement des mots.',
        ]
    else:
        lines.append("")
        lines.append("Tu ne peux déclencher AUCUNE action : utilise toujours \"none\".")

    lines += [
        "",
        'Ajoute TOUJOURS un champ "emotion" décrivant ton état, parmi exactement :',
        "neutre, colere, peur, joie, tristesse, mefiance (sans accent). Il sert à teinter ta voix.",
        "",
        "FORMAT DE RÉPONSE — réponds STRICTEMENT avec cet objet JSON, rien d'autre :",
        '{"speech": "ta réplique parlée", "emotion": "neutre", "action": {"type": "none"}}',
        'Pour une action : {"speech": "...", "emotion": "joie", "action": {"type": "follow", "target": "player"}}',
        'Pour donner un objet : {"speech": "...", "emotion": "neutre", "action": {"type": "give_item", "item": "bandage", "count": 1}}',
        'Pour donner de l\'argent : {"speech": "...", "emotion": "peur", "action": {"type": "give_money", "amount": 20}}',
    ]
    return "\n".join(lines)


def build_system_prompt(npc: dict, player: dict, world: dict, allowed: list[str]) -> str:
    """Prompt = [bloc fixe mis en cache] + [bloc variable : personnage, contexte]."""

    # ===== BLOC FIXE (préfixe stable -> cache Groq) =====
    lines = [_static_rules(allowed)]

    # ===== BLOC VARIABLE (change à chaque PNJ / contexte -> jamais en cache) =====
    lines += [
        "",
        "----",
        "TON PERSONNAGE :",
        f"Tu incarnes {npc.get('name', 'un habitant')}, {npc.get('job', 'habitant de Los Santos')}.",
        f"Personnalité : {npc.get('personality', 'ordinaire, neutre')}.",
        f"Ton : {npc.get('tone', 'naturel, familier')}.",
    ]
    if npc.get("knows"):
        lines.append(f"Ce que tu sais : {npc['knows']}.")
    if npc.get("ignores"):
        lines.append(f"Ce que tu ignores totalement : {npc['ignores']}. Si on t'en parle, dis que tu n'en sais rien.")

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

    if world.get("threatened"):
        lines += [
            "",
            "⚠ ON TE MENACE : le joueur pointe une arme droit sur toi, MAINTENANT.",
            "Tu as peur. Réagis de façon crédible : supplie, cède à ses demandes (tu peux lui",
            "donner de l'argent avec l'action give_money), lâche une info s'il en réclame, ou",
            "tente de fuir (action flee) si tu es courageux ou acculé. N'ignore jamais l'arme.",
        ]

    lines += [
        "",
        "Reste strictement dans la peau de ce personnage, et réponds au format JSON demandé plus haut.",
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
    emotion = "neutre"

    if isinstance(data, dict):
        raw_speech = data.get("speech")
        if isinstance(raw_speech, str):
            speech = raw_speech.strip()

        raw_emotion = _norm(str(data.get("emotion", "")))
        if raw_emotion in EMOTIONS:
            emotion = raw_emotion

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
        speech = _fallback_speech()
        action = {"type": "none"}
        emotion = "neutre"

    if len(speech) > MAX_SPEECH_CHARS:
        speech = speech[:MAX_SPEECH_CHARS].rsplit(" ", 1)[0] + "..."

    return {"speech": speech, "action": action, "emotion": emotion}


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

    async def _complete(json_mode: bool):
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        # Tous les fournisseurs ne gèrent pas le mode JSON strict (OpenRouter route
        # vers des back-ends variés) : on sait s'en passer, _parse_json est tolérant.
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return await client.chat.completions.create(**kwargs)

    try:
        resp = await _complete(True)
    except Exception as exc:
        # Le quota (429) doit remonter : main.py coupe la conversation proprement.
        if is_quota_error(exc):
            raise
        print(f"[llm] Mode JSON refusé ({exc}) — nouvel essai sans response_format")
        try:
            resp = await _complete(False)
        except Exception as exc2:
            if is_quota_error(exc2):
                raise
            print(f"[llm] Erreur LLM : {exc2}")
            return {"speech": _fallback_speech(), "action": {"type": "none"}, "emotion": "neutre"}

    raw = resp.choices[0].message.content

    return _sanitize(_parse_json(raw), allowed)
