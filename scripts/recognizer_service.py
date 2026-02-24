import os
import time
import threading
import cv2
import numpy as np
from fastapi import FastAPI
import uvicorn
from insightface.app import FaceAnalysis
import onnxruntime as ort

DB_DIR = "db"
CAM_INDEX = 0
W, H = 1280, 720
DET_SIZE = (640, 640)

# Conservative defaults (tune later)
THRESH_NAME = 0.60
THRESH_FAMILIAR = 0.40
MIN_DET_SCORE = 0.60
FRAME_FLUSH_READS = 3

app_web = FastAPI(title="face-recognizer")

def load_db(db_dir: str):
    gallery = {}  # name -> mean embedding
    for fn in os.listdir(db_dir):
        if fn.endswith(".npz"):
            name = fn[:-4]
            data = np.load(os.path.join(db_dir, fn))
            embs = data["embs"].astype(np.float32)
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            mean = embs.mean(axis=0)
            mean = mean / np.linalg.norm(mean)
            gallery[name] = mean
    return gallery

def cos_sim(a, b):
    return float(np.dot(a, b))

gallery = load_db(DB_DIR)
if not gallery:
    raise SystemExit("DB is empty. Run enroll.py first.")

available = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
print("ONNX providers:", available)
print("Using providers:", providers)

face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    raise SystemExit("Could not open camera")
cap_lock = threading.Lock()

def identify_once():
    # Flush a few buffered frames so each request uses a fresh image.
    with cap_lock:
        ret, frame = False, None
        for _ in range(FRAME_FLUSH_READS):
            ret, frame = cap.read()
    if not ret:
        return {"status": "camera_error", "name": None, "score": 0.0}

    faces = face_app.get(frame)
    if not faces:
        return {"status": "no_face", "name": None, "score": 0.0}

    # largest face only (reception behavior)
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    if float(getattr(f, "det_score", 0.0)) < MIN_DET_SCORE:
        return {"status": "low_conf_face", "name": None, "score": float(getattr(f, "det_score", 0.0))}

    x1, y1, x2, y2 = map(int, f.bbox)
    area = (x2 - x1) * (y2 - y1)
    if area < 140 * 140:
        return {"status": "too_far", "name": None, "score": 0.0}

    emb = f.embedding.astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    best_name, best_score = None, -1.0
    for name, mean_emb in gallery.items():
        s = cos_sim(emb, mean_emb)
        if s > best_score:
            best_score = s
            best_name = name

    if best_score >= THRESH_NAME:
        return {"status": "known", "name": best_name, "score": best_score}
    elif best_score >= THRESH_FAMILIAR:
        return {"status": "familiar", "name": None, "score": best_score}
    else:
        return {"status": "unknown", "name": None, "score": best_score}

@app_web.get("/identify")
def identify():
    return identify_once()

@app_web.get("/recognize")
def recognize_alias():
    return identify_once()

@app_web.get("/recognizer")
def recognizer_alias():
    return identify_once()

@app_web.get("/health")
def health():
    return {"ok": True, "gallery_size": len(gallery), "ts": time.time()}

@app_web.get("/")
def index():
    return {
        "ok": True,
        "service": "face-recognizer",
        "endpoints": ["/identify", "/recognize", "/recognizer", "/health"],
    }

@app_web.on_event("shutdown")
def on_shutdown():
    cap.release()

if __name__ == "__main__":
    # Local only, no external access
    uvicorn.run(app_web, host="127.0.0.1", port=8008)
