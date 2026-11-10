"""Riquadri dei volti di una o piu immagini (inference_server_venv: insightface + onnxruntime su CPU, antelopev2 gia sul disco; mai la GPU).
Ingresso (stdin JSON): {"images": [paths]} -> riga `@@JSON@@{"res": [[{"bbox": [x0, y0, x1, y1], "score": s}, ...], ...]}` (una lista per immagine, volti piu grandi per primi)."""
import json
import os
import sys

ROOT = os.environ.get("AC_INSIGHTFACE_ROOT", r"E:\AI\Models\insightface")
DETECTION_SIZE = 1024                    # i volti dell'Integra sono piccoli (30-70 px su 1500): a 640 verrebbero ridotti sotto i 20 px


def main():
    job = json.loads(sys.stdin.read())
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", root=ROOT, providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(DETECTION_SIZE, DETECTION_SIZE))
    res = []
    for path in job["images"]:
        image = cv2.imread(path)
        faces = app.get(image) if image is not None else []
        faces = sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        res.append([{"bbox": [round(float(v), 1) for v in f.bbox], "score": round(float(f.det_score), 3)} for f in faces])
    print("@@JSON@@" + json.dumps({"res": res}))


if __name__ == "__main__":
    main()
