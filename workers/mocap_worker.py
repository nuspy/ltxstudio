"""Worker mocap single-webcam (gira nel venv isolato `mocap_env`).

Cattura da UNA webcam, stima la posa 3D con MediaPipe Pose (monoculare, niente
calibrazione) e, allo stop, scrive un BVH umanoide (nomi ossa compatibili col
retarget: Hips/Spine/Spine1/Neck/Head, Left/Right Shoulder/Arm/ForeArm/Hand,
Left/Right UpLeg/Leg/Foot). Protocollo JSON-stdio con main process (mocap_service):

  stdin  : {"cmd":"start","camera":0,"out":"<bvh>","fps":30,"max_seconds":0}
           {"cmd":"stop"} {"cmd":"quit"}
  stdout : {"event":"ready"} {"event":"frame","preview":"<png>","n":N}
           {"event":"done","bvh":"<path>","frames":N} {"event":"error","msg":"..."}

NB: monoculare ⇒ profondità approssimata (jitter/drift). Per qualità superiore
usare l'import di una sessione FreeMoCap. Questo writer è volutamente semplice e
va validato sull'hardware reale.
"""
from __future__ import annotations

import json
import sys
import threading
import time

import numpy as np

# Scheletro: (nome, parent_idx, mp_joint_idx, mp_child_idx_per_direzione)
# Indici MediaPipe Pose: 0 naso, 11/12 spalle L/R, 13/14 gomiti, 15/16 polsi,
# 23/24 anche, 25/26 ginocchia, 27/28 caviglie.
SKELETON = [
    ("Hips", -1, None, None),          # 0  radice (mid anche)
    ("Spine", 0, None, None),          # 1  mid anche→mid spalle (1/2)
    ("Spine1", 1, None, None),         # 2  mid spalle
    ("Neck", 2, None, None),           # 3
    ("Head", 3, 0, None),              # 4  naso
    ("LeftShoulder", 2, 11, 13),       # 5
    ("LeftArm", 5, 11, 13),            # 6
    ("LeftForeArm", 6, 13, 15),        # 7
    ("LeftHand", 7, 15, None),         # 8
    ("RightShoulder", 2, 12, 14),      # 9
    ("RightArm", 9, 12, 14),           # 10
    ("RightForeArm", 10, 14, 16),      # 11
    ("RightHand", 11, 16, None),       # 12
    ("LeftUpLeg", 0, 23, 25),          # 13
    ("LeftLeg", 13, 25, 27),           # 14
    ("LeftFoot", 14, 27, None),        # 15
    ("RightUpLeg", 0, 24, 26),         # 16
    ("RightLeg", 16, 26, 28),          # 17
    ("RightFoot", 17, 28, None),       # 18
]


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _joint_world(lm, idx_def) -> np.ndarray:
    """Posizione 3D di un giunto dello scheletro dai landmark MediaPipe (metri)."""
    name, parent, mp_idx, _ = idx_def
    P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float64)
    if name == "Hips":
        return (P(23) + P(24)) / 2.0
    if name == "Spine1":
        return (P(11) + P(12)) / 2.0
    if name == "Spine":
        return ((P(23) + P(24)) / 2.0 + (P(11) + P(12)) / 2.0) / 2.0
    if name == "Neck":
        return (P(11) + P(12)) / 2.0 * 0.5 + P(0) * 0.5
    if mp_idx is not None:
        return P(mp_idx)
    return (P(23) + P(24)) / 2.0


def _quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternione (w,x,y,z) che ruota il versore a sul versore b."""
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    d = float(np.dot(a, b))
    if d > 0.9999:
        return np.array([1.0, 0, 0, 0])
    if d < -0.9999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b); axis /= (np.linalg.norm(axis) + 1e-9)
    ang = np.arccos(d)
    s = np.sin(ang / 2)
    return np.array([np.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_to_euler_zxy(q: np.ndarray) -> tuple:
    """Quaternione → Euler (Z,X,Y) gradi, ordine canale BVH ZXY."""
    w, x, y, z = q
    # rotazione → matrice
    xx, yy, zz = x * x, y * y, z * z
    r = np.array([
        [1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)],
    ])
    sx = -r[1, 2]
    sx = max(-1.0, min(1.0, sx))
    rx = np.arcsin(sx)
    if abs(sx) < 0.9999:
        rz = np.arctan2(r[1, 0], r[1, 1])
        ry = np.arctan2(r[0, 2], r[2, 2])
    else:
        rz = np.arctan2(-r[0, 1], r[0, 0]); ry = 0.0
    return (np.degrees(rz), np.degrees(rx), np.degrees(ry))


def _write_bvh(path: str, frames: list, fps: int) -> int:
    """Scrive un BVH dai frame di landmark (lista di liste di 33 landmark)."""
    n = len(SKELETON)
    # offset di riposo = vettore medio osso→figlio (dal primo frame valido)
    pos0 = np.array([_joint_world(frames[0], SKELETON[i]) for i in range(n)])
    offsets = np.zeros((n, 3))
    for i, (_, parent, _, _) in enumerate(SKELETON):
        offsets[i] = (pos0[i] - pos0[parent]) if parent >= 0 else np.array([0, 0, 0])
    scale = 100.0   # metri → cm (scala BVH leggibile)

    def children(i):
        return [j for j, (_, p, _, _) in enumerate(SKELETON) if p == i]

    lines = []

    def hier(i, indent):
        name = SKELETON[i][0]
        pad = "  " * indent
        head = "ROOT" if i == 0 else "JOINT"
        lines.append(f"{pad}{head} {name}")
        lines.append(f"{pad}{{")
        off = offsets[i] * scale
        lines.append(f"{pad}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
        if i == 0:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        ch = children(i)
        if ch:
            for c in ch:
                hier(c, indent + 1)
        else:
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET 0.0 0.0 5.0")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    lines.append("HIERARCHY")
    hier(0, 0)
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {1.0 / max(1, fps):.6f}")

    for lm in frames:
        pos = np.array([_joint_world(lm, SKELETON[i]) for i in range(n)])
        # rotazione world di ogni osso = allinea offset di riposo alla direzione corrente
        world_q = [np.array([1.0, 0, 0, 0])] * n
        for i, (_, parent, _, _) in enumerate(SKELETON):
            ch = children(i)
            if ch:
                cur_dir = pos[ch[0]] - pos[i]
                rest_dir = offsets[ch[0]]
                world_q[i] = _quat_between(rest_dir, cur_dir)
        row = []
        root = pos[0] * scale
        row += [f"{root[0]:.4f}", f"{root[1]:.4f}", f"{root[2]:.4f}"]
        for i in range(n):
            parent = SKELETON[i][1]
            if parent >= 0:
                pq = world_q[parent]
                # local = parent^-1 * world
                pw, px, py, pz = pq
                pinv = np.array([pw, -px, -py, -pz])
                lq = _qmul(pinv, world_q[i])
            else:
                lq = world_q[i]
            rz, rx, ry = _quat_to_euler_zxy(lq)
            row += [f"{rz:.4f}", f"{rx:.4f}", f"{ry:.4f}"]
        lines.append(" ".join(row))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(frames)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class Capture:
    def __init__(self):
        self._stop = threading.Event()
        self._frames = []

    def run(self, camera: int, out: str, fps: int, preview_path: str, max_seconds: float):
        import cv2
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True)
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            emit({"event": "error", "msg": f"webcam {camera} non disponibile"}); return
        emit({"event": "ready"})
        t0 = time.time(); n = 0; last_prev = 0.0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                self._frames.append(res.pose_world_landmarks.landmark)
                n += 1
            now = time.time()
            if now - last_prev > 0.1:   # ~10 anteprime/s
                try:
                    cv2.imwrite(preview_path, cv2.resize(frame, (320, 240)))
                    emit({"event": "frame", "preview": preview_path, "n": n})
                except Exception:
                    pass
                last_prev = now
            if max_seconds and (now - t0) >= max_seconds:
                break
        cap.release(); pose.close()
        if len(self._frames) < 2:
            emit({"event": "error", "msg": "troppo pochi frame catturati"}); return
        try:
            frames = _write_bvh(out, self._frames, fps)
            emit({"event": "done", "bvh": out, "frames": frames})
        except Exception as e:  # noqa: BLE001
            emit({"event": "error", "msg": f"scrittura BVH fallita: {e}"})

    def stop(self):
        self._stop.set()


def main() -> int:
    cap = None
    thread = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        if cmd == "start":
            cap = Capture()
            thread = threading.Thread(target=cap.run, kwargs=dict(
                camera=int(req.get("camera", 0)), out=req["out"],
                fps=int(req.get("fps", 30)),
                preview_path=req.get("preview", "mocap_preview.png"),
                max_seconds=float(req.get("max_seconds", 0))), daemon=True)
            thread.start()
        elif cmd == "stop":
            if cap:
                cap.stop()
            if thread:
                thread.join(timeout=30)
        elif cmd == "quit":
            if cap:
                cap.stop()
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
