"""Riferimenti nominati `@image_N` — UNICA convenzione dell'app per citare le immagini di riferimento nel prompt.

L'utente carica le immagini; ognuna riceve AL CARICAMENTO un nome stabile (`image_1`, `image_2`, …) e nel prompt le cita
con `@image_N` («la tartaruga in @image_1 mangia la lattuga in @image_2»). Alla generazione i token vengono tradotti nella
sintassi del MODELLO scelto (tabella `MODEL_SYNTAX`) e le immagini passate nell'ordine giusto.

Regole:
 - i nomi sono STABILI: togliere image_2 non rinumera le altre; una nuova immagine prende il primo numero libero;
 - il prefisso (`image`/`video`/`audio`, minuscolo) è UNICO per tutte le lingue e NON si traduce (deciso il 29/10); il
   parser accetta in più gli alias dei vecchi nomi (`@Foto_N`, `@Video_N`, `@Audio_N`), che la GUI non mostra più;
 - i prompt vecchi con `<Picture N>` / `<Video N>` / `<Audio N>` (tab Video) si leggono ancora e vengono tradotti
   come gli `@image_N`; la GUI però mostra e inserisce solo `@image_N`;
 - un token che cita un riferimento inesistente è un ERRORE da segnalare prima di generare, mai ignorato in silenzio.

Modulo puro (nessun Qt): testabile con pytest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# nome per tipo di riferimento. `PREFIX` = ciò che l'utente vede e digita; `ALIASES` = ciò che il parser accetta.
# Prefisso UNICO per tutte le lingue (deciso dall'utente il 29/10): minuscolo, non tradotto. Gli alias (case-insensitive)
# sono i vecchi nomi: `@Foto_N`, `@Video_N`, `@Audio_N` si leggono e si traducono, ma la GUI mostra solo i nuovi.
PREFIX = {"image": "image", "video": "video", "audio": "audio"}
ALIASES = {"image": ["image", "Foto"], "video": ["video"], "audio": ["audio"]}
KINDS = ("image", "video", "audio")


def display_prefix(kind: str = "image") -> str:
    return PREFIX[kind]


def name_of(kind: str, n: int) -> str:
    """Nome visibile: `image_3`."""
    return f"{PREFIX[kind]}_{n}"


def token_of(kind: str, n: int) -> str:
    """Token nel prompt: `@image_3`."""
    return "@" + name_of(kind, n)


def _alias_regex() -> re.Pattern:
    alts = []
    for kind in KINDS:
        for a in ALIASES[kind]:
            alts.append((a, kind))
    # ordina per lunghezza decrescente: un alias non deve mangiare il prefisso di un altro
    alts.sort(key=lambda t: -len(t[0]))
    return re.compile(r"@(?P<alias>" + "|".join(re.escape(a) for a, _ in alts) + r")_(?P<n>\d+)(?![\w])",
                      re.IGNORECASE)


_LEGACY = re.compile(r"<(?P<alias>Picture|Video|Audio)\s+(?P<n>\d+)>", re.IGNORECASE)
_LEGACY_KIND = {"picture": "image", "video": "video", "audio": "audio"}


def _kind_of_alias(alias: str) -> str | None:
    al = alias.lower()
    for kind in KINDS:
        if any(al == a.lower() for a in ALIASES[kind]):
            return kind
    return None


@dataclass(frozen=True)
class Token:
    kind: str
    n: int
    start: int
    end: int
    legacy: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.kind, self.n)


def scan(prompt: str) -> list[Token]:
    """Tutti i token (`@image_N`, e i legacy `<Picture N>`) nel prompt, in ordine di apparizione."""
    out: list[Token] = []
    for m in _alias_regex().finditer(prompt or ""):
        kind = _kind_of_alias(m.group("alias"))
        if kind:
            out.append(Token(kind, int(m.group("n")), m.start(), m.end(), False))
    for m in _LEGACY.finditer(prompt or ""):
        out.append(Token(_LEGACY_KIND[m.group("alias").lower()], int(m.group("n")), m.start(), m.end(), True))
    out.sort(key=lambda t: t.start)
    return out


def unknown_tokens(prompt: str, existing) -> list[Token]:
    """Token del prompt che citano un riferimento che NON esiste. `existing` = insieme di (kind, n)."""
    have = set(existing)
    return [t for t in scan(prompt) if t.key not in have]


def uncited(prompt: str, existing) -> list[tuple[str, int]]:
    """Riferimenti caricati ma non citati nel prompt (vengono passati comunque al modello)."""
    cited = {t.key for t in scan(prompt)}
    return [k for k in sorted(set(existing), key=lambda k: (KINDS.index(k[0]), k[1])) if k not in cited]


class NameRegistry:
    """Numerazione STABILE per tipo: `assign()` dà il primo numero libero, `release()` lo libera senza rinumerare."""

    def __init__(self) -> None:
        self._used: dict[str, set[int]] = {k: set() for k in KINDS}

    def assign(self, kind: str) -> int:
        used = self._used[kind]
        n = 1
        while n in used:
            n += 1
        used.add(n)
        return n

    def reserve(self, kind: str, n: int) -> None:
        self._used[kind].add(int(n))

    def release(self, kind: str, n: int) -> None:
        self._used[kind].discard(int(n))

    def clear(self) -> None:
        for s in self._used.values():
            s.clear()


# ── sintassi per modello ───────────────────────────────────────────────────────────────────────────────────
# `fmt` = come il MODELLO cita l'immagine k-esima passata (k = posizione 1-based nell'elenco inviato al worker).
# `source` dice DA DOVE viene la convenzione; `verified` = provata sul codice/live (True) o solo dedotta (False).
MODEL_SYNTAX: dict[str, dict] = {
    "qwen21": {
        "fmt": {"image": "image {n}"}, "max_images": 10, "verified": True,
        "source": "core/qwen21_spec.py MASK_HINT («…highlighted in red in image 1…») e workers/qwen_image21_infer.py: "
                  "le immagini passate come lista, nell'ordine dato, sono image 1…N",
    },
    "krea2_edit": {
        "fmt": {"image": "image {n}"}, "max_images": 2, "verified": False,
        "source": "workers/krea2_edit_pipeline.py: le immagini entrano come blocchi <|vision_start|> PRIMA del prompt, "
                  "nell'ordine dato; il template NON ha etichette «Picture N». «image N» è la citazione posizionale "
                  "in linguaggio naturale (non provata live)",
    },
    "flux2_klein": {
        "fmt": {"image": "image {n}"}, "max_images": 4, "verified": False,
        "source": "inference_server/backends/image_flux.py + diffusers Flux2KleinPipeline: `image` e' una LISTA, ogni "
                  "riferimento ha latenti e id di posizione propri (nessuna etichetta nel template). «image N» e' la "
                  "citazione posizionale in linguaggio naturale; il massimo (4) e' una proposta prudente (VRAM), NON provato live",
    },
    "kontext_stitch": {
        "fmt": {"image": "the {ord} picture from the left"}, "max_images": 4, "verified": False,
        "source": "ui/main_window.generate_compose: FLUX.1 Kontext vede UNA immagine, le foto affiancate da sinistra a destra: «image 2» non "
                  "significa nulla per lui, la posizione si. NON provato dal vivo (revisione 31/10: la donna non veniva portata nella scena)",
    },
    "flux3": {
        "fmt": {"image": "the {ord} image"}, "max_images": 10, "verified": False,
        "source": "docs diffusers (pipelines/flux3) e system prompt dell'upsampler FLUX.3 (task 13, 18/10): i riferimenti si "
                  "citano per ORDINALE («the first image», «the second image»…) nell'ordine di invio; nel caption "
                  "l'upsampler li chiama <ref_image_0>…; i riquadri di edit usano `ref_image_N` zero-based "
                  "(core/flux3_caption.ref_name). NON provato dal vivo",
    },
    "h3": {
        "fmt": {"image": "<Picture {n}>", "video": "<Video {n}>", "audio": "<Audio {n}>"}, "verified": True,
        "source": "core/h3_prompt.py (prompt strutturato MiniMax H3: <Picture N>/<Video N>/<Audio N>)",
    },
    "ltx": {
        "fmt": {"image": "Image {n}", "video": "Video {n}", "audio": "Audio {n}"}, "verified": True,
        "source": "memoria reference-panel-multi: validato live 17/08 con la LoRA MSR (con «<Picture N>» il riferimento "
                  "umano veniva ignorato)",
    },
}


_ORD = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def format_ref(fmt: str, k: int) -> str:
    """Sintassi del modello per il k-esimo riferimento (1-based): `{n}` = numero, `{ord}` = ordinale inglese (FLUX.3)."""
    return fmt.format(n=k, ord=_ORD[k - 1] if 1 <= k <= len(_ORD) else f"{k}th")


def translate(prompt: str, order: dict[str, list[int]], style: str) -> str:
    """Sostituisce ogni token (`@image_N` o legacy) con la sintassi di `style` per l'immagine k-esima.

    `order[kind]` = numeri dei riferimenti NELL'ORDINE in cui vengono passati al worker (es. {'image': [1, 3]}:
    image_3 è la 2ª immagine → «image 2»). I token di riferimenti inesistenti restano com'erano (li intercetta
    `unknown_tokens` prima di generare)."""
    fmt = MODEL_SYNTAX[style]["fmt"]
    pos = {kind: {n: i + 1 for i, n in enumerate(ns)} for kind, ns in order.items()}
    out, last = [], 0
    for t in scan(prompt):
        k = pos.get(t.kind, {}).get(t.n)
        f = fmt.get(t.kind)
        if k is None or f is None:
            continue
        out.append(prompt[last:t.start])
        out.append(format_ref(f, k))
        last = t.end
    out.append(prompt[last:])
    return "".join(out)
