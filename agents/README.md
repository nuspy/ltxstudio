# LTX Studio — pacchetti agente (plugin + skill)

Espone la **REST API locale** di LTX Studio a diversi agenti/CLI. Un solo core (l'API su
`http://127.0.0.1:8200`), adattatori per formato.

## Contratti macchina (comuni a tutti)
- `openapi.json` — contratto OpenAPI completo (importabile da qualsiasi client/agent).
- `tools.openai.json` — tool *function-calling* (OpenAI / Codex / compatibili).
- `manifest.json` — catalogo neutro delle capacità (`name`, `method`, `path`, `group`, `params`).

Rigenerali dopo ogni modifica all'API con:
```
python -m agents.build_packs
```

## Agenti supportati (in `agents/<agente>/`)
| Agente | Cartella | Note |
|---|---|---|
| **Hermes** (default) | `hermes/` | `SKILL.md` + `plugin.json` |
| OpenClaw | `openclaw/` | idem |
| Claude Code | `claude_code/` | idem; skill installabile in `~/.claude/skills/ltxstudio` |
| Google Antigravity CLI | `antigravity/` | idem |
| OpenAI Codex | `openai_codex/` | usa `tools.openai.json` per il function-calling |
| Generico | `generic/` | qualsiasi client via OpenAPI |

Ogni cartella contiene:
- `SKILL.md` — guida operativa (autostart, job asincroni, download link, regole SFW/NSFW, catena path).
- `plugin.json` — manifest del plugin (base URL, riferimento a `openapi.json`, comando di autostart).

## Ciclo d'uso (tutti gli agenti)
1. **Autostart**: `python -m api.autostart` → `READY http://127.0.0.1:8200` (o `GET /health`).
   L'API **si spegne da sola dopo 30 min** di inattività: riavviala al bisogno.
2. **Crea job**: `POST /<gruppo>/<azione>` → `{job_id}`.
3. **Polling**: `GET /jobs/{job_id}` finché `success`/`error`.
4. **Scarica**: usa `files[].download_url` (`GET /download?path=...`).
5. Coda **seriale** (un job GPU alla volta); file utente via `POST /files/upload`; concatena i path.

## Gruppi di capacità (32 endpoint)
immagini (8) · video (10) · musica/audio (6) · 3D/rig/mondo/convert (4) · film (3) · infra (upload).
