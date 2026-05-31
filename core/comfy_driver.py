"""Client per pilotare ComfyUI da LTX Studio.

REST: stdlib (urllib) — zero dipendenze, endpoint verificati in server.py:
  POST /prompt, GET /history/{id}, GET /view, GET /object_info,
  POST /upload/image, POST /upload/mask, POST /interrupt, GET /system_stats.
WebSocket: websocket-client (opzionale) per progress + preview binaria live
  (BinaryEventTypes.PREVIEW_IMAGE: 4 byte tipo + bytes immagine JPEG/PNG).

Il WS gira in un thread; espone callback (on_progress, on_preview, on_executed,
on_status) che la UI Qt collega ai propri segnali tramite preview_bus.
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Callable

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188


@dataclass
class ComfyConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @property
    def http(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"


class ComfyDriver:
    def __init__(self, config: ComfyConfig | None = None):
        self.cfg = config or ComfyConfig()
        self.client_id = uuid.uuid4().hex
        self._ws = None
        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        # callback impostate dalla UI
        self.on_status: Callable[[dict], None] | None = None
        self.on_progress: Callable[[int, int], None] | None = None
        self.on_executing: Callable[[str | None], None] | None = None
        self.on_executed: Callable[[str, dict], None] | None = None
        self.on_preview: Callable[[bytes], None] | None = None

    # ---------- REST (stdlib) ----------
    def _get(self, path: str, params: dict | None = None) -> bytes:
        url = self.cfg.http + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()

    def _get_json(self, path: str, params: dict | None = None):
        return json.loads(self._get(path, params).decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.http + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return json.loads(body.decode("utf-8")) if body else {}

    def is_up(self) -> bool:
        try:
            self._get_json("/system_stats")
            return True
        except Exception:
            return False

    def object_info(self, node_class: str | None = None) -> dict:
        return self._get_json(f"/object_info/{node_class}" if node_class else "/object_info")

    def gpu_vram_gb(self) -> float | None:
        """Ritorna la VRAM totale (GB) del primo device GPU di ComfyUI. None se non disponibile."""
        try:
            stats = self._get_json("/system_stats")
            devs = stats.get("devices") or []
            for d in devs:
                # cuda/rocm/xpu — saltiamo CPU
                if (d.get("type") or "").lower() in ("cuda", "rocm", "xpu", "mps"):
                    vt = d.get("vram_total")
                    if vt:
                        return float(vt) / (1024 ** 3)
            # fallback: primo device qualunque
            if devs and devs[0].get("vram_total"):
                return float(devs[0]["vram_total"]) / (1024 ** 3)
        except Exception:
            pass
        return None

    def models_in(self, folder: str) -> list[str]:
        """Elenco file modello per cartella (checkpoints, loras, vae, diffusion_models, ...)."""
        try:
            return self._get_json(f"/models/{folder}")
        except Exception:
            return []

    def submit(self, prompt: dict, extra_data: dict | None = None) -> str:
        """Invia il prompt-API. Ritorna prompt_id."""
        payload = {"prompt": prompt, "client_id": self.client_id}
        if extra_data:
            payload["extra_data"] = extra_data
        resp = self._post_json("/prompt", payload)
        if resp.get("node_errors"):
            raise RuntimeError(f"node_errors: {resp['node_errors']}")
        return resp["prompt_id"]

    def history(self, prompt_id: str) -> dict:
        return self._get_json(f"/history/{prompt_id}")

    def interrupt(self) -> None:
        try:
            self._post_json("/interrupt", {})
        except Exception:
            pass

    def free_memory(self) -> bool:
        """Chiede a ComfyUI di scaricare i modelli dalla VRAM e liberare la memoria.
        Ritorna True se la richiesta è andata a buon fine."""
        try:
            self._post_json("/free", {"unload_models": True, "free_memory": True})
            return True
        except Exception:
            return False

    def view_url(self, filename: str, subfolder: str = "", type_: str = "output") -> str:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        return f"{self.cfg.http}/view?{q}"

    def download_view(self, filename: str, subfolder: str = "", type_: str = "output") -> bytes:
        return self._get("/view", {"filename": filename, "subfolder": subfolder, "type": type_})

    def upload_image(self, filepath: str, subfolder: str = "", overwrite: bool = True) -> dict:
        """Upload multipart su /upload/image. Usato per immagini, maschere-base e audio."""
        return self._upload("/upload/image", filepath, subfolder, overwrite)

    def upload_mask(self, mask_path: str, ref_filename: str, ref_subfolder: str = "",
                    ref_type: str = "input") -> dict:
        """Upload maschera: il server fa alpha-merge sull'immagine di riferimento."""
        extra = {"original_ref": json.dumps(
            {"filename": ref_filename, "subfolder": ref_subfolder, "type": ref_type})}
        return self._upload("/upload/mask", mask_path, "", True, extra_fields=extra)

    def _upload(self, path: str, filepath: str, subfolder: str, overwrite: bool,
                extra_fields: dict | None = None) -> dict:
        boundary = "----ltxstudio" + uuid.uuid4().hex
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        fname = filepath.replace("\\", "/").split("/")[-1]
        ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        parts: list[bytes] = []

        def field(name, value):
            parts.append(
                (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode())

        if subfolder:
            field("subfolder", subfolder)
        field("overwrite", "true" if overwrite else "false")
        for k, v in (extra_fields or {}).items():
            field(k, v)
        parts.append(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fname}\"\r\n"
             f"Content-Type: {ctype}\r\n\r\n").encode())
        body = b"".join(parts) + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.cfg.http + path, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    # ---------- WebSocket (progress + preview) ----------
    def start_ws(self) -> None:
        try:
            import websocket  # websocket-client
        except ImportError as e:
            raise RuntimeError(
                "Manca 'websocket-client'. Installa: pip install websocket-client") from e

        url = f"{self.cfg.ws}?clientId={self.client_id}"

        def run():
            while not self._ws_stop.is_set():
                try:
                    self._ws = websocket.create_connection(url, timeout=10)
                    while not self._ws_stop.is_set():
                        msg = self._ws.recv()
                        if isinstance(msg, bytes):
                            self._handle_binary(msg)
                        else:
                            self._handle_text(msg)
                except Exception:
                    if self._ws_stop.is_set():
                        break
                    time.sleep(2)  # reconnect

        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=run, daemon=True)
        self._ws_thread.start()

    def stop_ws(self) -> None:
        self._ws_stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def _handle_text(self, msg: str) -> None:
        try:
            data = json.loads(msg)
        except Exception:
            return
        t = data.get("type")
        d = data.get("data", {})
        if t == "status" and self.on_status:
            self.on_status(d)
        elif t == "progress" and self.on_progress:
            self.on_progress(int(d.get("value", 0)), int(d.get("max", 0)))
        elif t == "executing" and self.on_executing:
            self.on_executing(d.get("node"))
        elif t == "executed" and self.on_executed:
            self.on_executed(d.get("node", ""), d.get("output", {}))

    def _handle_binary(self, msg: bytes) -> None:
        # primi 4 byte = tipo evento (1=JPEG,2=PNG); poi 4 byte formato; poi immagine
        if len(msg) < 8 or self.on_preview is None:
            return
        # header: event_type (4) + image_format (4), poi i dati immagine
        self.on_preview(msg[8:])


if __name__ == "__main__":
    drv = ComfyDriver()
    up = drv.is_up()
    print(f"ComfyUI raggiungibile su {drv.cfg.http}: {up}")
    if up:
        oi = drv.object_info("UNETLoader")
        print("UNETLoader presente:", "UNETLoader" in oi)
        print("diffusion_models:", drv.models_in("diffusion_models")[:5])
