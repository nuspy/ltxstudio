"""Tavolo di co-scrittura interattivo (modalità «Interattivo» del Copione). Invece di erogare un trattamento
in un colpo solo, l'LLM propone una struttura a BEAT (i battiti drammaturgici) e, per ogni beat, alcuni BIVI
(rami alternativi). L'autore SCEGLIE un ramo per beat; poi l'LLM intreccia le scelte in un racconto coerente.

È il gesto che mancava: un LLM che RILANCIA («e se invece…?») invece di produrre e basta.

- propose_beats(idea, styles, genres, n) -> [{"beat": titolo, "options": [{"label","text"}]}]
- assemble(idea, chosen, styles, genres) -> racconto cinematografico (prosa) che realizza i bivi scelti
"""
from __future__ import annotations


_BEATS_SYS = (
    "You are a co-writer at a writers' table. From the user's seed idea, propose a beat structure for the story "
    "(the key dramatic beats). For EACH beat give 2-3 DISTINCT branching options ('what if…') the author can "
    "choose between — genuinely different directions, not rewordings. Answer in the SAME language as the idea. "
    "Output STRICT JSON: {\"beats\":[{\"beat\":str,\"options\":[{\"label\":str,\"text\":str}]}]}. "
    "beat = short title of the dramatic beat; label = 3-6 word tag of the branch; text = 1-2 sentences describing "
    "that branch. Keep @Name references if present. No prose outside JSON."
)

_ASSEMBLE_SYS = (
    "You are a screenwriter. The author picked one branch per dramatic beat at the writers' table. Weave the "
    "chosen branches into a single coherent CINEMATIC PROSE treatment (a 'racconto') in the SAME language: "
    "evocative prose of what we see and feel, honoring every chosen branch in order. Keep @Name references. "
    "Output only the treatment, no headings or JSON."
)


def _meta(styles: str, genres: str) -> str:
    bits = []
    if styles:
        bits.append(f"Visual style: {styles}")
    if genres:
        bits.append(f"Genre(s): {genres}")
    return ("\n" + ". ".join(bits) + ".") if bits else ""


def propose_beats(idea: str, styles: str = "", genres: str = "", n: int = 5) -> list[dict]:
    """[{beat, options:[{label, text}]}] — struttura a beat con bivi. [] se LLM non disponibile/parse fallito."""
    from core import llm_service as L
    if not (idea and idea.strip() and L.available()):
        return []
    user = f"Seed idea: {idea.strip()}\nAim for about {max(3, int(n))} beats.{_meta(styles, genres)}"
    try:
        data = L.complete_json(_BEATS_SYS, user, max_tokens=1800)
    except Exception:                                       # noqa: BLE001
        return []
    beats = []
    for b in (data.get("beats") or []):
        opts = [{"label": (o.get("label") or "").strip(), "text": (o.get("text") or "").strip()}
                for o in (b.get("options") or []) if isinstance(o, dict) and (o.get("text") or o.get("label"))]
        if b.get("beat") and opts:
            beats.append({"beat": (b.get("beat") or "").strip(), "options": opts})
    return beats


def assemble(idea: str, chosen: list[dict], styles: str = "", genres: str = "") -> str:
    """Intreccia i rami scelti in un racconto. `chosen` = [{beat, label, text}] in ordine. '' se non fattibile."""
    from core import llm_service as L
    if not (chosen and L.available()):
        return ""
    picks = "\n".join(f"- {c.get('beat', '')}: {c.get('label', '')} — {c.get('text', '')}" for c in chosen)
    user = f"Seed idea: {(idea or '').strip()}\nChosen branches (in order):\n{picks}{_meta(styles, genres)}"
    try:
        return L.complete(_ASSEMBLE_SYS, user, max_tokens=1600).strip()
    except Exception:                                       # noqa: BLE001
        return ""
