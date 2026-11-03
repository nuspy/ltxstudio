"""Worker volti dei controlli automatici (inference_server_venv: insightface + onnxruntime, provider CPU, antelopev2 gia sul disco).
Ingresso (stdin JSON): {"ref": path, "images": [paths]} -> riga `@@JSON@@{"res": [{"n": volti, "max": somiglianza massima}]}`."""
import json
import os
import sys

ROOT = os.environ.get("AC_INSIGHTFACE_ROOT", r"E:\AI\Models\insightface")


def main():
    job = json.loads(sys.stdin.read())
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", root=ROOT, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    im = cv2.imread(job["ref"])
    fs = app.get(im) if im is not None else []
    if not fs:
        print("@@JSON@@" + json.dumps({"res": [{"n": 0, "max": -1.0, "errore": "nessun volto nella foto di riferimento"} for _ in job["images"]]}))
        return
    best = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    ref = best.normed_embedding

    def crop_hp(img, bb):
        """Volto ritagliato (bbox + 15 %), grigio 96x96, passa-alto: per confrontare i dettagli fini di due volti."""
        x1, y1, x2, y2 = [float(v) for v in bb]
        w, h = x2 - x1, y2 - y1
        c = img[max(0, int(y1 - .15 * h)):int(y2 + .15 * h), max(0, int(x1 - .15 * w)):int(x2 + .15 * w)]
        if c.size == 0:
            return None
        g = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        return g - cv2.GaussianBlur(g, (0, 0), 2.0)
    ref_hp = crop_hp(im, best.bbox)
    res = []
    for p in job["images"]:
        img = cv2.imread(p)
        faces = app.get(img) if img is not None else []
        sims = [float(np.dot(ref, f.normed_embedding)) for f in faces]
        r = {"n": len(faces), "max": round(max(sims), 3) if sims else -1.0}
        if faces:
            bf = faces[int(np.argmax(sims))]
            r["bbox"] = [round(float(v), 1) for v in bf.bbox]
            r["larghezza"] = round(float(bf.bbox[2] - bf.bbox[0]), 1)
            hp = crop_hp(img, bf.bbox)
            if hp is not None and ref_hp is not None and hp.std() > 1e-6 and ref_hp.std() > 1e-6:
                r["ncc_riferimento"] = round(float(np.corrcoef(hp.ravel(), ref_hp.ravel())[0, 1]), 3)   # ~1 = il volto e LA STESSA FOTO del riferimento
        res.append(r)
    print("@@JSON@@" + json.dumps({"res": res}))


if __name__ == "__main__":
    main()
