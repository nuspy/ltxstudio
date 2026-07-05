"""ATF Detector — app Android (Kivy) con FOTOCAMERA e CARICAMENTO foto (#107).

Verifica se una foto è stata generata con AI Tools Forge, leggendo la firma di provenienza invisibile
(+ testo custom). Puoi SCATTARE una foto (anche allo schermo di un altro dispositivo) o CARICARLA dalla
galleria. Il rilevamento gira in locale sul telefono (nessuna rete), tramite detector_stego.py (numpy+PIL).

Build APK: buildozer android debug   (vedi README.md). File entrypoint richiesto da buildozer: main.py.
"""
from __future__ import annotations

import os
import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

import detector_stego


class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=18, spacing=14, **kw)
        self.add_widget(Label(text="[b]🔎 ATF Detector[/b]", markup=True, font_size="26sp",
                              size_hint_y=None, height="52dp", color=(0.56, 0.42, 0.96, 1)))
        self.add_widget(Label(text="Verifica se una foto è stata creata con AI Tools Forge.",
                              size_hint_y=None, height="34dp"))

        b_cam = Button(text="📷  Scatta una foto", size_hint_y=None, height="64dp", font_size="20sp",
                       background_color=(0.56, 0.42, 0.96, 1))
        b_cam.bind(on_release=self._capture)
        self.add_widget(b_cam)

        b_up = Button(text="🖼  Carica una foto", size_hint_y=None, height="64dp", font_size="20sp")
        b_up.bind(on_release=self._upload)
        self.add_widget(b_up)

        self.result = Label(text="", markup=True, halign="center", valign="middle", font_size="18sp")
        self.result.bind(size=lambda *_: setattr(self.result, "text_size", self.result.size))
        self.add_widget(self.result)

    # ---- input ----
    def _capture(self, *_):
        try:
            from plyer import camera
            path = os.path.join(App.get_running_app().user_data_dir, "atf_capture.jpg")
            self._set_status("Apertura fotocamera…")
            camera.take_picture(filename=path, on_complete=self._on_photo)
        except Exception as e:                               # noqa: BLE001
            self._set_status(f"[color=ff5555]Fotocamera non disponibile: {e}[/color]")

    def _upload(self, *_):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=self._on_selection,
                                  filters=[["Immagini", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]])
        except Exception as e:                               # noqa: BLE001
            self._set_status(f"[color=ff5555]Selezione file non disponibile: {e}[/color]")

    def _on_photo(self, path):
        if path and os.path.exists(path):
            self._analyze(path)

    def _on_selection(self, selection):
        if selection:
            self._analyze(selection[0])

    # ---- analisi (in thread) ----
    def _analyze(self, path):
        self._set_status("Analisi in corso… (può richiedere qualche secondo)")
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        try:
            found, z, text = detector_stego.detect_path(path)
        except Exception as e:                               # noqa: BLE001
            Clock.schedule_once(lambda *_: self._set_status(f"[color=ff5555]Errore: {e}[/color]"))
            return
        Clock.schedule_once(lambda *_: self._show(found, z, text))

    def _show(self, found, z, text):
        if found:
            msg = "[color=2f9e44][b]✅ GENERATO con AI Tools Forge[/b][/color]"
            if text:
                msg += f'\n\nTesto firma: [b]"{text}"[/b]'
            msg += f"\n\nAffidabilità: {z:.1f}"
        else:
            msg = "[color=b0b4c0][b]❌ Nessuna firma rilevata[/b][/color]\n\nNon risulta creato con AI Tools Forge."
        self.result.text = msg

    def _set_status(self, t):
        self.result.text = t


class ATFDetectorApp(App):
    def build(self):
        self.title = "ATF Detector"
        return Root()


if __name__ == "__main__":
    ATFDetectorApp().run()
