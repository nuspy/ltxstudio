"""Visore 3D interattivo (three.js dentro QWebEngineView) con selezione,
cancellazione e separazione di sub-oggetti.

A differenza di `_Model3DViewer` (trellis_tab.py, basato su <model-viewer>, di
sola visualizzazione), questo visore:
- carica un GLB con three.js (GLTFLoader + OrbitControls + Raycaster);
- al click evidenzia il sub-oggetto e ne riporta il nome a Python via QWebChannel;
- offre pulsanti "Separa parti sciolte" / "Elimina selezione" (operazioni PERSISTENTI
  delegate a Blender via main_window) e Import/Export.

Le operazioni mesh persistenti NON le fa il visore: emette `request_mesh_op`
(op=separate_loose|delete, names) verso main_window, che lancia Blender
(scripts/blender/mesh_ops.py) e poi RICARICA il file editato qui con `load_model`.

Se QtWebEngine non è disponibile, degrada a un messaggio (come il viewer storico).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QObject, QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox,
)

from ui.i18n_manager import tr

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebChannel import QWebChannel
    _HAS_WEBENGINE = True
except Exception:  # pragma: no cover - dipende dall'installazione PySide6
    _HAS_WEBENGINE = False


class _SelectionBridge(QObject):
    """Ponte JS→Python: il visore three.js riporta il nome del sub-oggetto cliccato."""
    selectionChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self._name = ""

    @Slot(str)
    def selected(self, name: str):
        self._name = name or ""
        self.selectionChanged.emit(self._name)

    @property
    def name(self) -> str:
        return self._name


class InteractiveModel3DViewer(QWidget):
    """Visore 3D con selezione/cancellazione/separazione e Import/Export."""

    request_mesh_op = Signal(dict)   # {"op": "separate_loose"|"delete", "names": [...]}
    request_import = Signal()         # l'utente vuole caricare un file dallo stadio
    request_export = Signal()         # l'utente vuole esportare l'asset corrente

    def __init__(self):
        super().__init__()
        self._glb_path: str = ""
        self._selected: str = ""
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)

        if _HAS_WEBENGINE:
            self.view = QWebEngineView()
            s = self.view.settings()
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            s.setAttribute(QWebEngineSettings.WebGLEnabled, True)
            self._bridge = _SelectionBridge()
            self._bridge.selectionChanged.connect(self._on_selection)
            self._channel = QWebChannel()
            self._channel.registerObject("bridge", self._bridge)
            self.view.page().setWebChannel(self._channel)
            v.addWidget(self.view, 1)
            self._show_placeholder()
        else:
            self.view = None
            lbl = QLabel(tr("trellis.viewer.no_webengine"))
            lbl.setWordWrap(True); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#9aa1b0; padding:24px;")
            v.addWidget(lbl, 1)

        # etichetta selezione
        self.sel_lbl = QLabel("")
        self.sel_lbl.setStyleSheet("color:#74c0fc; padding:2px 6px;")
        v.addWidget(self.sel_lbl)

        # barra pulsanti
        bar = QHBoxLayout()
        self.btn_separate = QPushButton(tr("viewer.separate"))
        self.btn_separate.setToolTip(tr("viewer.separate.tt"))
        self.btn_separate.clicked.connect(self._on_separate)
        self.btn_delete = QPushButton(tr("viewer.delete_sel"))
        self.btn_delete.setToolTip(tr("viewer.delete_sel.tt"))
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_import = QPushButton(tr("viewer.import"))
        self.btn_import.clicked.connect(self.request_import.emit)
        self.btn_export = QPushButton(tr("viewer.export"))
        self.btn_export.clicked.connect(self.request_export.emit)
        self.btn_folder = QPushButton(tr("trellis.viewer.open_folder"))
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_saveas = QPushButton(tr("trellis.viewer.save_as"))
        self.btn_saveas.clicked.connect(self._save_as)
        for b in (self.btn_separate, self.btn_delete, self.btn_import,
                  self.btn_export, self.btn_folder, self.btn_saveas):
            bar.addWidget(b)
        bar.addStretch()
        v.addLayout(bar)
        self._update_buttons()

    # ---------- API compatibile col viewer storico ----------
    def load_model(self, glb_path: str):
        self._glb_path = glb_path
        self._selected = ""
        self.sel_lbl.setText("")
        self._update_buttons()
        if not _HAS_WEBENGINE or self.view is None:
            return
        p = Path(glb_path)
        if not p.exists():
            self._show_placeholder(tr("trellis.viewer.not_found"))
            return
        html = _VIEWER_HTML.replace("__SRC__", p.name)
        base = QUrl.fromLocalFile(str(p.parent) + os.sep)
        self.view.setHtml(html, base)

    def show_result(self, glb_path: str):
        self.load_model(glb_path)

    def get_edited_path(self) -> str:
        return self._glb_path

    # ---------- selezione ----------
    def _on_selection(self, name: str):
        self._selected = name
        self.sel_lbl.setText(tr("viewer.selected").format(name=name) if name else "")
        self._update_buttons()

    def _update_buttons(self):
        has = bool(self._glb_path)
        for b in (self.btn_separate, self.btn_export, self.btn_folder, self.btn_saveas):
            b.setEnabled(has)
        self.btn_delete.setEnabled(has and bool(self._selected))

    # ---------- operazioni mesh (delegate a Blender via main_window) ----------
    def _on_separate(self):
        if self._glb_path:
            self.request_mesh_op.emit({"op": "separate_loose", "input": self._glb_path})

    def _on_delete(self):
        if self._glb_path and self._selected:
            self.request_mesh_op.emit(
                {"op": "delete", "input": self._glb_path, "names": [self._selected]})

    # ---------- pulsanti file ----------
    def _open_folder(self):
        if self._glb_path:
            try:
                os.startfile(str(Path(self._glb_path).parent))  # noqa: S606 (Windows)
            except Exception:
                pass

    def _save_as(self):
        if not self._glb_path or not Path(self._glb_path).exists():
            return
        ext = Path(self._glb_path).suffix.lstrip(".") or "glb"
        dest, _ = QFileDialog.getSaveFileName(
            self, tr("trellis.viewer.save_as"),
            str(Path.home() / Path(self._glb_path).name), f"3D model (*.{ext})")
        if dest:
            try:
                shutil.copyfile(self._glb_path, dest)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, tr("trellis.viewer.save_as"), str(e))

    def _show_placeholder(self, msg: str = ""):
        if not _HAS_WEBENGINE or self.view is None:
            return
        text = msg or tr("trellis.viewer.placeholder")
        self.view.setHtml(_PLACEHOLDER_HTML.replace("__MSG__", text))


# ── three.js: scena + GLTFLoader + OrbitControls + Raycaster + QWebChannel ──────
_VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#1e1f26;overflow:hidden}
#c{width:100%;height:100vh;display:block}
#hint{position:fixed;left:8px;bottom:6px;color:#6b7280;font:12px sans-serif}</style>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script type="importmap">{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}</script>
</head><body>
<canvas id="c"></canvas>
<div id="hint">click = seleziona · trascina = ruota · rotella = zoom</div>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let bridge = null;
new QWebChannel(qt.webChannelTransport, (ch) => { bridge = ch.objects.bridge; });

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1e1f26);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.4); dir.position.set(3,5,4); scene.add(dir);

let root = null, selected = null;
const HL = new THREE.Color(0xff7043);
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();

function frame(obj){
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const c = box.getCenter(new THREE.Vector3());
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(0, size*0.25, size*1.2));
  camera.near = size/100; camera.far = size*100; camera.updateProjectionMatrix();
  controls.update();
}

// nome del sub-oggetto = primo discendente diretto di root che contiene il mesh
function topName(o){ let n=o; while(n.parent && n.parent!==root) n=n.parent; return n.name||o.name||""; }
function rootObjFor(o){ let n=o; while(n.parent && n.parent!==root) n=n.parent; return n; }

function clearHL(){ if(selected){ selected.traverse(m=>{ if(m.material&&m._hl){ m.material.emissive&&m.material.emissive.copy(m._oe); m._hl=false; }}); selected=null; } }
function highlight(obj){
  clearHL();
  obj.traverse(m=>{ if(m.isMesh && m.material && m.material.emissive){ m._oe=m.material.emissive.clone(); m.material.emissive.copy(HL); m._hl=true; }});
  selected = obj;
}

renderer.domElement.addEventListener('pointerdown', (e)=>{
  if(!root) return;
  const r = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX-r.left)/r.width)*2-1;
  mouse.y = -((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObject(root, true);
  if(hits.length){ const obj = rootObjFor(hits[0].object); highlight(obj);
    if(bridge) bridge.selected(topName(hits[0].object)); }
  else { clearHL(); if(bridge) bridge.selected(""); }
});

const loader = new GLTFLoader();
loader.load('__SRC__', (gltf)=>{ root = gltf.scene; scene.add(root); frame(root); },
  undefined, (err)=>{ document.body.innerHTML='<div style="color:#9aa1b0;font:14px sans-serif;padding:40px;text-align:center">Errore di caricamento del modello 3D.<br>'+err+'</div>'; });

window.addEventListener('resize', ()=>{ camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); });
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script>
</body></html>"""

_PLACEHOLDER_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#1e1f26;display:flex;
align-items:center;justify-content:center}
#m{color:#9aa1b0;font-family:sans-serif;text-align:center;padding:40px;max-width:80%}</style>
</head><body><div id="m">__MSG__</div></body></html>"""
