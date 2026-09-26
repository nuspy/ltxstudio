"""Genera i pacchetti agente per la REST API di LTX Studio a partire dall'OpenAPI live.

Produce (in agents/):
  - openapi.json          : contratto OpenAPI completo (ogni agente può importarlo)
  - manifest.json         : catalogo neutro delle capacità (name, method, path, group, params)
  - tools.openai.json     : tool function-calling (OpenAI / Codex / compatibili)
  - <agente>/SKILL.md     : guida operativa per ogni agente (Hermes default + altri)
  - <agente>/plugin.json  : manifest del plugin (dove pertinente)

Uso:  python -m agents.build_packs
"""
from __future__ import annotations

import json
from pathlib import Path

from api.app import app
from api.config import BASE_URL

HERE = Path(__file__).resolve().parent

AGENTS = {
    "hermes":      {"title": "Hermes", "default": True},
    "openclaw":    {"title": "OpenClaw", "default": False},
    "claude_code": {"title": "Claude Code", "default": False},
    "antigravity": {"title": "Google Antigravity CLI", "default": False},
    "openai_codex": {"title": "OpenAI Codex", "default": False},
    "generic":     {"title": "Generico (OpenAPI)", "default": False},
}


def _resolve_schema(spec: dict, node: dict) -> dict:
    """Inline superficiale di un $ref di un requestBody (properties + required)."""
    if "$ref" in node:
        ref = node["$ref"].split("/")[-1]
        return spec.get("components", {}).get("schemas", {}).get(ref, {})
    return node


def build_manifest(spec: dict) -> list[dict]:
    caps = []
    for path, methods in spec.get("paths", {}).items():
        post = methods.get("post")
        if not post:
            continue
        group = (post.get("tags") or ["misc"])[0]
        schema = {}
        rb = post.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
        if rb:
            schema = _resolve_schema(spec, rb)
        caps.append({
            "name": path.strip("/").replace("/", "_"),
            "method": "POST", "path": path, "group": group,
            "summary": post.get("summary", ""),
            "params": schema.get("properties", {}),
            "required": schema.get("required", []),
        })
    return caps


def build_openai_tools(caps: list[dict]) -> list[dict]:
    tools = []
    for c in caps:
        tools.append({
            "type": "function",
            "function": {
                "name": c["name"],
                "description": f"{c['summary']} (POST {c['path']} → job asincrono; poi GET /jobs/{{job_id}}).",
                "parameters": {"type": "object", "properties": c["params"], "required": c["required"]},
            },
        })
    return tools


SKILL_DESCRIPTION = (
    "Pilota la REST API LOCALE di LTX Studio per CREARE e MODIFICARE immagini, video, film/videoclip, "
    "musica, modelli 3D, rig/skinning e animazioni. Job asincroni che ritornano link di download. "
    "Copre testo/immagine/audio->immagine e video (t2v/i2v/a2v, inpaint, kontext, upscale, lip-sync, "
    "controlnet, extend), canzoni (HeartMuLa/ACE-Step), TTS/voce, immagine->3D (TRELLIS), rig, mondo 3D, "
    "e shot di film. Usa quando l'utente vuole generare/modificare foto, video, musica, film o modelli 3D "
    "da QUALSIASI input. L'API si avvia da sola e si spegne dopo 30 min di inattivita."
)


SKILL_BODY = """\
Guidi la generazione/modifica di **immagini, video, film, videoclip, musica, modelli 3D, rig/skinning e
animazioni** tramite la **REST API locale di LTX Studio** su `{base}` (Swagger `{base}/docs`).

## 0. Avvio automatico e spegnimento
- **Prima di ogni sessione**, assicura che l'API sia attiva: esegui `python -m api.autostart`
  (ritorna `READY {base}` quando pronta) oppure `GET {base}/health`. Se `health` non risponde,
  l'API è spenta: **avviala** con l'autostart, non fallire.
- L'API **si spegne da sola dopo 30 minuti** di inattività (nessuna richiesta e nessun job). È normale:
  al prossimo uso rilancia l'autostart. Non tenerla viva artificialmente.

## 1. Modello di esecuzione (ASINCRONO)
1. Ogni azione è un `POST` che ritorna `{{ "job_id": "...", "status": "pending" }}`.
2. **Fai polling** di `GET {base}/jobs/{{job_id}}` finché `status` = `success` o `error`.
   - `status=running` include `progress` (0–1) e `message`.
   - a `success` la risposta ha `files: [{{ path, name, download_url }}]`.
   - a `error` la risposta ha `error` (spesso indica un prerequisito da installare).
3. **Consegna** all'utente i `download_url` (`GET {base}/download?path=...`).
4. **Un solo job GPU alla volta**: la coda è seriale. Non lanciare job in parallelo; attendi il precedente.

## 2. File in ingresso
- Carica i file dell'utente UNA volta con `POST {base}/files/upload` (multipart, campo `file`) →
  ritorna `{{ path }}`. Riusa quel `path` nei campi `*_path` (`image_path`, `video_path`,
  `audio_path`, `mask_path`, `mesh_path`, ...).
- **Concatenamento**: l'output di un job (`files[0].path`) è un path locale sul server: passalo
  direttamente come input del passo successivo, senza download/upload intermedi.

## 3. Scelta dell'endpoint (intento → chiamata)
- descrizione → immagine: `POST /image/generate` (`arch` flux/flux2/zimage/ernie, `model`)
- modifica foto (istruzione): `POST /image/edit`; colore: `/image/recolor`; area precisa:
  `/image/inpaint` (serve `mask_path`; creala con `/image/segment`); estendi bordi: `/image/outpaint`
- upscale/restauro foto: `/image/upscale`; volto da foto su un'altra: `/image/faceswap`;
  identità del volto in generazione: `image/generate` con `pulid_face`
- descrizione → video: `POST /video/generate` (t2v). foto → video: stesso con `image_path` (i2v).
  audio → video / canto: stesso con `audio_path` (a2v)
- keyframe a tempo: `/video/keyframes`; prolunga: `/video/extend`; rigenera sezione: `/video/retake`;
  modifica da istruzione/stile: `/video/v2v_edit`; copia movimenti: `/video/controlnet`
- lip-sync (foto+voce): `/video/lipsync` (`separate_vocals` per isolare la voce, `talk_region` mouth/face)
- avatar parlante / filmato parlato da un testo: `POST /avatar/generate` (LongCat 1.5 o MiniMax H3; immagine da
  testo/foto/personaggio, voce TTS/dialogo con marcatori `[pause:500ms]` `[emotion:happy]…[/emotion]`
  `[speed:1.2]` `[accel:0.9-1.3]` `[pitch:+2]` `[volume:-6]` `[whisper]` `[emph]`); opzioni installate
  `GET /avatar/options`. Guida completa + client: skill **ltx-avatar-film** (`skills/ltx-avatar-film`)
- qualità video: `/video/enhance` (SeedVR2/FlashVSR), +fps `/video/rife`, volto `/video/faceswap`
- musica/canzone: `/music/song` (engine heartmula|acestep); sonorizza video: `/music/sonify`;
  videoclip canzone→video: `/music/music_video`; voce TTS: `/audio/tts`; trascrivi: `/audio/transcribe`
- 3D da foto: `/threed/image_to_3d`; scheletro+skinning: `/threed/rig`; scena 3D: `/threed/world`;
  converti mesh: `/threed/convert`
- film a inquadrature: `/film/script` (trattamento), `/film/asset_image` (personaggi/keyframe),
  `/film/shot_video` (shot dello storyboard)

## 4. Regole
- **SFW/NSFW**: default SFW. Per contenuti adulti chiedi conferma esplicita UNA volta; non dedurre il
  consenso dal prompt.
- **Selezione modello = compito tuo**: proponi default sensati; chiedi all'utente solo il necessario.
- **Prerequisiti**: alcuni endpoint (SeedVR2, FlashVSR, TRELLIS, WorldGen, HeartMuLa, ACE-Step, MMAudio,
  TTS clone, face-swap, PuLID) richiedono venv/pesi installati. Se un job torna `error` con "non
  installato/preflight", spiega all'utente come installarlo (script in `scripts/install_*.py`).
- **Job lunghi** (video/film/musica = minuti): annuncia l'attesa, fai polling con backoff (2–5 s),
  non ripetere il POST.

## 5. Contratto macchina
- OpenAPI completo: `{base}/openapi.json` (o `agents/openapi.json`).
- Tool function-calling: `agents/tools.openai.json`. Catalogo neutro: `agents/manifest.json`.
"""


PLUGIN_JSON = {
    "name": "ltxstudio",
    "version": "1.0.0",
    "description": "Genera/modifica immagini, video, film, musica, 3D/rig/animazioni via LTX Studio (locale).",
    "api": {"base_url": BASE_URL, "openapi": "agents/openapi.json", "autostart": "python -m api.autostart"},
}


def main():
    spec = app.openapi()
    caps = build_manifest(spec)
    tools = build_openai_tools(caps)

    (HERE / "openapi.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "manifest.json").write_text(json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "tools.openai.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, meta in AGENTS.items():
        d = HERE / key
        d.mkdir(parents=True, exist_ok=True)
        default_note = " (agente di DEFAULT)" if meta["default"] else ""
        header = (f"---\nname: ltxstudio\n"
                  f"description: {SKILL_DESCRIPTION}\n"
                  f"target: {meta['title']}{default_note}\n---\n\n"
                  f"# LTX Studio — skill per {meta['title']}{default_note}\n\n")
        (d / "SKILL.md").write_text(header + SKILL_BODY.format(base=BASE_URL), encoding="utf-8")
        plugin = dict(PLUGIN_JSON, target=meta["title"], default=meta["default"])
        (d / "plugin.json").write_text(json.dumps(plugin, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK — {len(caps)} capacità · agenti: {', '.join(AGENTS)}")
    print(f"     openapi.json, manifest.json, tools.openai.json + <agente>/SKILL.md+plugin.json")


if __name__ == "__main__":
    main()
