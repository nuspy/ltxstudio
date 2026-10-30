"""OpenPose CORPO (18 keypoint COCO, stile CMU / lllyasviel) nativo: modello + rilevamento + disegno.

Sostituisce `controlnet_aux.OpenposeDetector` (non installato) e ComfyUI. Il modello e' la rete CMU a due rami
(PAF 38 canali + heatmap 19 canali, 6 stadi) i cui pesi stanno in `body_pose_model.pth` (184 tensori, chiavi
`conv1_1.weight`, `Mconv7_stage6_L2.bias`, ...). Pure torch + numpy + cv2 + scipy.

API:
  BodyPoseModel                                   # nn.Module, forward(x) -> (paf[N,38,h,w], heat[N,19,h,w])
  load_model(path=None, device="cpu")             # carica i pesi STRICT, eval()
  detect(image_rgb, model=None, device="cpu", scale_search=(0.5,), ...)
        -> {"candidate": [[x,y,score,id]...], "subset": [[18 id + score_tot + n_parti]...]}   (x,y in pixel)
  draw_bodypose(canvas, candidate, subset)        # scheletro colorato (17 limb + 18 punti), canvas RGB uint8
  pose_image(image_rgb, ...) -> (canvas_rgb, result)   # canvas nero + scheletro; nero (+ warning) se nessuno

Regole: nessun contesto CUDA nel processo GUI -> device="cpu" di default (cuda solo se il chiamante lo passa).
Input RGB (il modello CMU vuole BGR: la conversione e' interna).
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

WEIGHTS_PATH = r"E:\AI\Models\ControlNet\Openpose\body_pose_model.pth"
WEIGHTS_NAME = "body_pose_model.pth"
WEIGHTS_SIZE = 209267595          # byte del file CMU body-18 verificato (184 tensori)

# 19 coppie di keypoint (indici 1-based COCO-18 di CMU) e canali PAF corrispondenti (19..56)
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
MAP_IDX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
           [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
KEYPOINT_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "r_hip",
                  "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]


# --------------------------------------------------------------------------------------------- modello
class BodyPoseModel(nn.Module):
    """Rete CMU body-18. Nomi dei layer = chiavi dello state_dict originale (caricamento strict)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        def conv(name, cin, cout, k, pad):
            self.add_module(name, nn.Conv2d(cin, cout, k, 1, pad))

        # backbone VGG (block0)
        self._block0 = []
        for name, cin, cout, pool_after in [
            ("conv1_1", 3, 64, False), ("conv1_2", 64, 64, True),
            ("conv2_1", 64, 128, False), ("conv2_2", 128, 128, True),
            ("conv3_1", 128, 256, False), ("conv3_2", 256, 256, False),
            ("conv3_3", 256, 256, False), ("conv3_4", 256, 256, True),
            ("conv4_1", 256, 512, False), ("conv4_2", 512, 512, False),
            ("conv4_3_CPM", 512, 256, False), ("conv4_4_CPM", 256, 128, False),
        ]:
            conv(name, cin, cout, 3, 1)
            self._block0.append((name, pool_after))

        # stadio 1: due rami (L1 = PAF 38, L2 = heatmap 19)
        self._stage1 = {}
        for j, nout in ((1, 38), (2, 19)):
            seq = []
            for n in (1, 2, 3):
                conv(f"conv5_{n}_CPM_L{j}", 128, 128, 3, 1)
                seq.append((f"conv5_{n}_CPM_L{j}", True))
            conv(f"conv5_4_CPM_L{j}", 128, 512, 1, 0)
            seq.append((f"conv5_4_CPM_L{j}", True))
            conv(f"conv5_5_CPM_L{j}", 512, nout, 1, 0)
            seq.append((f"conv5_5_CPM_L{j}", False))      # ultimo layer senza ReLU
            self._stage1[j] = seq

        # stadi 2..6: 5 conv 7x7 + 2 conv 1x1, input = concat(PAF, heat, feature) = 185 canali
        self._stages = {}
        for s in range(2, 7):
            for j, nout in ((1, 38), (2, 19)):
                seq = []
                conv(f"Mconv1_stage{s}_L{j}", 185, 128, 7, 3)
                seq.append((f"Mconv1_stage{s}_L{j}", True))
                for n in (2, 3, 4, 5):
                    conv(f"Mconv{n}_stage{s}_L{j}", 128, 128, 7, 3)
                    seq.append((f"Mconv{n}_stage{s}_L{j}", True))
                conv(f"Mconv6_stage{s}_L{j}", 128, 128, 1, 0)
                seq.append((f"Mconv6_stage{s}_L{j}", True))
                conv(f"Mconv7_stage{s}_L{j}", 128, nout, 1, 0)
                seq.append((f"Mconv7_stage{s}_L{j}", False))
                self._stages[(s, j)] = seq

    def _run(self, x, seq):
        for name, act in seq:
            x = getattr(self, name)(x)
            if act:
                x = self.relu(x)
        return x

    def forward(self, x):
        for name, pool_after in self._block0:
            x = self.relu(getattr(self, name)(x))
            if pool_after:
                x = self.pool(x)
        feat = x
        paf = self._run(feat, self._stage1[1])
        heat = self._run(feat, self._stage1[2])
        for s in range(2, 7):
            cat = torch.cat([paf, heat, feat], 1)
            paf = self._run(cat, self._stages[(s, 1)])
            heat = self._run(cat, self._stages[(s, 2)])
        return paf, heat


def check_weights(path: str | None = None) -> str:
    """Ritorna il percorso dei pesi o solleva RuntimeError CHIARO (nome file + dimensione attesa)."""
    p = path or WEIGHTS_PATH
    if not os.path.isfile(p):
        raise RuntimeError(f"Pesi OpenPose assenti: {WEIGHTS_NAME} (~{WEIGHTS_SIZE / 2**20:.0f} MB) non trovato in "
                           f"{p}: scaricalo dal tab «Pesi modelli» o copialo in quella cartella.")
    return p


def load_model(path: str | None = None, device: str = "cpu"):
    """Crea il modello e carica i pesi STRICT (nessuna chiave mancante/in eccesso). Ritorna il modello in eval()."""
    p = check_weights(path)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    model = BodyPoseModel()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------------------------- rilevamento
def _pad_right_down(img, stride, pad_value):
    h, w = img.shape[:2]
    pad = [0, 0, 0 if (h % stride == 0) else stride - (h % stride),
           0 if (w % stride == 0) else stride - (w % stride)]
    out = img
    if pad[2]:
        out = np.concatenate([out, np.full((pad[2], w, out.shape[2]), pad_value, dtype=out.dtype)], axis=0)
    if pad[3]:
        out = np.concatenate([out, np.full((out.shape[0], pad[3], out.shape[2]), pad_value, dtype=out.dtype)], axis=1)
    return out, pad


def _heat_paf(model, bgr, device, scale_search, boxsize=368, stride=8, pad_value=128):
    """Heatmap (H,W,19) e PAF (H,W,38) mediate sulle scale, alla risoluzione dell'immagine originale."""
    import cv2
    H, W = bgr.shape[:2]
    heat_avg = np.zeros((H, W, 19), dtype=np.float32)
    paf_avg = np.zeros((H, W, 38), dtype=np.float32)
    mult = [s * boxsize / H for s in scale_search]
    for scale in mult:
        test = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        padded, pad = _pad_right_down(test, stride, pad_value)
        im = np.ascontiguousarray(np.transpose(padded.astype(np.float32)[:, :, :, None], (3, 2, 0, 1))) / 256 - 0.5
        data = torch.from_numpy(im).to(device)
        with torch.inference_mode():
            paf_t, heat_t = model(data)
        heat = np.transpose(heat_t.squeeze(0).float().cpu().numpy(), (1, 2, 0))
        paf = np.transpose(paf_t.squeeze(0).float().cpu().numpy(), (1, 2, 0))
        heat = cv2.resize(heat, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heat = heat[:padded.shape[0] - pad[2], :padded.shape[1] - pad[3], :]
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:padded.shape[0] - pad[2], :padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (W, H), interpolation=cv2.INTER_CUBIC)
        heat_avg += heat / len(mult)
        paf_avg += paf / len(mult)
    return heat_avg, paf_avg


def _find_peaks(heat_avg, thre1):
    from scipy.ndimage import gaussian_filter
    all_peaks, counter = [], 0
    for part in range(18):
        map_ori = heat_avg[:, :, part]
        one = gaussian_filter(map_ori, sigma=3)
        left = np.zeros(one.shape); left[1:, :] = one[:-1, :]
        right = np.zeros(one.shape); right[:-1, :] = one[1:, :]
        up = np.zeros(one.shape); up[:, 1:] = one[:, :-1]
        down = np.zeros(one.shape); down[:, :-1] = one[:, 1:]
        binary = np.logical_and.reduce((one >= left, one >= right, one >= up, one >= down, one > thre1))
        ys, xs = np.nonzero(binary)
        peaks = [(int(x), int(y), float(map_ori[y, x]), counter + i) for i, (x, y) in enumerate(zip(xs, ys))]
        all_peaks.append(peaks)
        counter += len(peaks)
    return all_peaks


def _connect_limbs(all_peaks, paf_avg, img_h, thre2, mid_num=10):
    connection_all, special_k = [], []
    for k in range(len(MAP_IDX)):
        score_mid = paf_avg[:, :, [x - 19 for x in MAP_IDX[k]]]
        cand_a = all_peaks[LIMB_SEQ[k][0] - 1]
        cand_b = all_peaks[LIMB_SEQ[k][1] - 1]
        n_a, n_b = len(cand_a), len(cand_b)
        if n_a == 0 or n_b == 0:
            special_k.append(k)
            connection_all.append([])
            continue
        cands = []
        for i in range(n_a):
            for j in range(n_b):
                vec = np.subtract(cand_b[j][:2], cand_a[i][:2]).astype(np.float64)
                norm = max(0.001, math.sqrt(vec[0] ** 2 + vec[1] ** 2))
                vec = vec / norm
                xs = np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num)
                ys = np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)
                xi = np.round(xs).astype(int); yi = np.round(ys).astype(int)
                vec_x = score_mid[yi, xi, 0]
                vec_y = score_mid[yi, xi, 1]
                mid_scores = vec_x * vec[0] + vec_y * vec[1]
                s_prior = float(mid_scores.sum() / len(mid_scores) + min(0.5 * img_h / norm - 1, 0))
                crit1 = len(np.nonzero(mid_scores > thre2)[0]) > 0.8 * len(mid_scores)
                crit2 = s_prior > 0
                if crit1 and crit2:
                    cands.append([i, j, s_prior, s_prior + cand_a[i][2] + cand_b[j][2]])
        cands = sorted(cands, key=lambda x: x[2], reverse=True)
        connection = np.zeros((0, 5))
        for c in range(len(cands)):
            i, j, s = cands[c][0:3]
            if i not in connection[:, 3] and j not in connection[:, 4]:
                connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                if len(connection) >= min(n_a, n_b):
                    break
        connection_all.append(connection)
    return connection_all, special_k


def _assemble(all_peaks, connection_all, special_k):
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sub in all_peaks for item in sub], dtype=np.float64).reshape(-1, 4)
    for k in range(len(MAP_IDX)):
        if k in special_k:
            continue
        part_as = connection_all[k][:, 0]
        part_bs = connection_all[k][:, 1]
        idx_a, idx_b = np.array(LIMB_SEQ[k]) - 1
        for i in range(len(connection_all[k])):
            found, subset_idx = 0, [-1, -1]
            for j in range(len(subset)):
                if subset[j][idx_a] == part_as[i] or subset[j][idx_b] == part_bs[i]:
                    subset_idx[found] = j
                    found += 1
            if found == 1:
                j = subset_idx[0]
                if subset[j][idx_b] != part_bs[i]:
                    subset[j][idx_b] = part_bs[i]
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[int(part_bs[i]), 2] + connection_all[k][i][2]
            elif found == 2:
                j1, j2 = subset_idx
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0:        # unione di due persone parziali
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, 0)
                else:                                               # conflitto: aggiungo solo il nuovo punto
                    subset[j1][idx_b] = part_bs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate[int(part_bs[i]), 2] + connection_all[k][i][2]
            elif not found and k < 17:
                row = -1 * np.ones(20)
                row[idx_a] = part_as[i]
                row[idx_b] = part_bs[i]
                row[-1] = 2
                row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack([subset, row])
    drop = [i for i in range(len(subset)) if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4]
    subset = np.delete(subset, drop, axis=0)
    return candidate, subset


def detect(image_rgb, model=None, device: str = "cpu", scale_search=(0.5,), thre1: float = 0.1,
           thre2: float = 0.05, weights_path: str | None = None) -> dict:
    """Rileva i corpi in `image_rgb` (HxWx3 uint8 RGB). Ritorna {"candidate": [[x,y,score,id]...], "subset": [...]}
    con x,y in pixel dell'immagine; subset[n][0..17] = id del candidato per ogni keypoint (-1 = assente),
    subset[n][18] = score totale, subset[n][19] = numero di parti. Nessuna persona -> liste vuote (nessuna eccezione)."""
    import cv2
    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_rgb deve essere HxWx3, ricevuto shape {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(img[:, :, ::-1])
    if model is None:
        model = load_model(weights_path, device)
    heat_avg, paf_avg = _heat_paf(model, bgr, device, scale_search)
    all_peaks = _find_peaks(heat_avg, thre1)
    conn, special = _connect_limbs(all_peaks, paf_avg, img.shape[0], thre2)
    candidate, subset = _assemble(all_peaks, conn, special)
    return {"candidate": candidate.tolist(), "subset": subset.tolist()}


# --------------------------------------------------------------------------------------------- disegno
def draw_bodypose(canvas, candidate, subset):
    """Disegna lo scheletro OpenPose standard su `canvas` (HxWx3 uint8, modificato e ritornato): 17 limb come
    ellissi piene (spessore 4) a colore pieno, poi attenuati x0.6, poi 18 punti (raggio 4) a colore pieno."""
    import cv2
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1, 4)
    subset = np.asarray(subset, dtype=np.float64).reshape(-1, 20)
    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(LIMB_SEQ[i]) - 1]
            if -1 in index:
                continue
            ys = candidate[index.astype(int), 0]
            xs = candidate[index.astype(int), 1]
            mx, my = np.mean(xs), np.mean(ys)
            length = ((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(xs[0] - xs[1], ys[0] - ys[1]))
            polygon = cv2.ellipse2Poly((int(my), int(mx)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, COLORS[i])
    canvas[:] = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, COLORS[i], thickness=-1)
    return canvas


def pose_image(image_rgb, model=None, device: str = "cpu", scale_search=(0.5,), weights_path: str | None = None):
    """Immagine di controllo openpose: scheletro colorato su fondo nero, stessa risoluzione dell'input.
    Ritorna (canvas_rgb uint8, result dict). Nessuna persona -> canvas nero + warning nel log."""
    img = np.asarray(image_rgb)
    res = detect(img, model=model, device=device, scale_search=scale_search, weights_path=weights_path)
    canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if not res["subset"]:
        log.warning("openpose: nessuna persona rilevata, restituisco un'immagine di controllo nera")
        return canvas, res
    draw_bodypose(canvas, res["candidate"], res["subset"])
    return canvas, res
