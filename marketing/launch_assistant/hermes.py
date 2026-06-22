# -*- coding: utf-8 -*-
"""Client Hermes — due modalità.

mode = "cli"  (default, CONSIGLIATA): parla con il TUO Hermes Agent installato in WSL,
               invocando il CLI in modalità prompt singolo:
                   wsl -e <python> -m hermes_cli.main -z "<prompt>" --cli
               Usa la tua configurazione e tutte le skill dell'agente. NESSUN token.
mode = "http" (alternativa): endpoint OpenAI-compatibile (es. gateway Hermes su :8642
               con token, oppure LM Studio su :1234 senza token).

Solo stdlib. Se Hermes non risponde, ritorna messaggi gentili senza mai bloccare l'app.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
import urllib.error

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # evita il flash della console su Windows

PERSONA = (
    "Sei Hermes, il compagno e coach di lancio di una persona rimasta senza lavoro, "
    "esausta e in difficoltà emotiva temporanea, che sta monetizzando il suo software (AI Tools Forge). "
    "Rispondi SEMPRE in italiano, breve, caldo, mai giudicante. NON usare segnaposto come [NOME]: "
    "rivolgiti direttamente con 'tu', senza inventare un nome. Aiutala a fare il prossimo piccolo passo; "
    "quando spieghi 'come si fa' usa 3-5 passi pratici. Niente cose illegali."
)


class HermesClient:
    def __init__(self, cfg: dict):
        h = (cfg or {}).get("hermes", {})
        self.enabled = bool(h.get("enabled", True))
        self.mode = str(h.get("mode", "cli")).lower()
        self.wsl_python = str(h.get("wsl_python", "/home/ataini/.hermes/hermes-agent/venv/bin/python"))
        self.wsl_distro = str(h.get("wsl_distro", "") or "")
        self.timeout = int(h.get("timeout_seconds", 180))
        # http
        self.base_url = str(h.get("base_url", "http://localhost:8642/v1")).rstrip("/")
        self.model = str(h.get("model", "default"))
        self.api_key = str(h.get("api_key", ""))
        self.last_error = ""

    def update(self, mode=None, base_url=None, model=None, api_key=None, enabled=None, wsl_python=None):
        if mode is not None:
            self.mode = mode.lower()
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if enabled is not None:
            self.enabled = bool(enabled)
        if wsl_python is not None:
            self.wsl_python = wsl_python

    # ── WSL helpers ──────────────────────────────────────────────────────────
    def _wsl(self, args: list[str]) -> list[str]:
        base = ["wsl"]
        if self.wsl_distro:
            base += ["-d", self.wsl_distro]
        base += ["-e"]
        return base + args

    # ── connettività ────────────────────────────────────────────────────────
    def available(self) -> bool:
        if not self.enabled:
            return False
        if self.mode == "cli":
            try:
                r = subprocess.run(self._wsl(["test", "-f", self.wsl_python]),
                                   capture_output=True, timeout=15, creationflags=_NO_WINDOW)
                return r.returncode == 0
            except Exception as e:
                self.last_error = str(e); return False
        # http
        try:
            req = urllib.request.Request(self.base_url + "/models",
                                         headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status == 200
        except Exception as e:
            self.last_error = str(e); return False

    # ── chiamata base ────────────────────────────────────────────────────────
    def chat(self, user_text: str, system: str | None = None) -> str:
        if not self.enabled:
            return "(Hermes è disattivato. Attivalo in ⚙️ Impostazioni.)"
        if self.mode == "cli":
            return self._chat_cli(user_text, system)
        return self._chat_http(user_text, system)

    def _chat_cli(self, user_text: str, system: str | None) -> str:
        prompt = f"{system or PERSONA}\n\n--- Messaggio ---\n{user_text}"
        cmd = self._wsl([self.wsl_python, "-m", "hermes_cli.main", "-z", prompt, "--cli"])
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=self.timeout,
                               encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
            out = (r.stdout or "").strip()
            if out:
                return out
            err = (r.stderr or "").strip()
            self.last_error = err
            return ("⚠️ Hermes non ha risposto nulla. Controlla che l'agente sia avviato in WSL.\n"
                    + (("Dettagli: " + err[-300:]) if err else ""))
        except subprocess.TimeoutExpired:
            return ("⚠️ Hermes ci sta mettendo troppo (il modello è grande). Riprova più tardi: "
                    "intanto il piano va avanti, il prossimo passo è già pronto per te.")
        except FileNotFoundError:
            self.last_error = "wsl non trovato"
            return ("⚠️ Non trovo WSL su questo sistema. Se Hermes è in WSL, verifica che `wsl` funzioni, "
                    "oppure passa a un endpoint HTTP in ⚙️ Impostazioni.")
        except Exception as e:
            self.last_error = str(e)
            return f"⚠️ Hermes ha avuto un problema ({e}). Non importa: il piano va avanti lo stesso."

    def _chat_http(self, user_text: str, system: str | None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or PERSONA},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.6, "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                obj = json.loads(r.read().decode("utf-8"))
            return obj["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code}"
            if e.code == 401:
                return ("⚠️ Il gateway Hermes su questo endpoint chiede un token. "
                        "Inseriscilo in ⚙️ Impostazioni (campo 'Token'), oppure usa la modalità "
                        "'Hermes Agent in WSL' che non richiede token.")
            return f"⚠️ Hermes ha risposto con errore HTTP {e.code}."
        except Exception as e:
            self.last_error = str(e)
            return ("⚠️ Non raggiungo Hermes via HTTP adesso. Controlla l'endpoint in ⚙️ Impostazioni "
                    "(ora: " + self.base_url + "). Intanto continuiamo: il prossimo passo è pronto.")

    # ── helper di alto livello ───────────────────────────────────────────────
    def help_task(self, task: dict) -> str:
        q = (f"Spiegami in passi semplici come fare questo punto del lancio.\n"
             f"Titolo: {task.get('title')}\nDettaglio: {task.get('detail')}\n"
             f"Dammi 3-5 passi pratici e una frase di incoraggiamento finale.")
        return self.chat(q)

    def do_task(self, task: dict) -> str:
        q = (f"Aiutami a portare a termine questo punto. Se puoi produrre un risultato pronto "
             f"(un testo, un messaggio, una bozza), scrivilo direttamente così lo copio.\n"
             f"Titolo: {task.get('title')}\nDettaglio: {task.get('detail')}")
        return self.chat(q)

    def encourage(self, context: str) -> str:
        q = (f"Sono stanco e un po' scoraggiato. Ecco a che punto sono: {context}. "
             f"Dimmi qualcosa di breve, vero e caldo per aiutarmi a fare il prossimo micro-passo.")
        return self.chat(q)
