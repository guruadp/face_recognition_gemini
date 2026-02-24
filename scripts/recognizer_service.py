import os
import time
import threading
import json
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from insightface.app import FaceAnalysis
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "db"))
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_INDICES_ENV = os.getenv("CAM_INDICES", "").strip()
W = int(os.getenv("CAM_WIDTH", "640"))
H = int(os.getenv("CAM_HEIGHT", "480"))
DET_EDGE = int(os.getenv("DET_EDGE", "320"))
DET_SIZE = (DET_EDGE, DET_EDGE)
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")

# Conservative defaults (tune later)
THRESH_NAME = 0.60
THRESH_FAMILIAR = 0.40
MIN_DET_SCORE = 0.60
FRAME_FLUSH_READS = int(os.getenv("FRAME_FLUSH_READS", "1"))
ENROLL_DEFAULT_SAMPLES = 20
ENROLL_MAX_FRAMES = 240

app_web = FastAPI(title="face-recognizer")


class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64, description="Person identifier")
    samples: int = Field(default=ENROLL_DEFAULT_SAMPLES, ge=5, le=50)

def load_db(db_dir: str):
    gallery = {}  # name -> mean embedding
    if not os.path.isdir(db_dir):
        return gallery
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
    print("[WARN] DB is empty. Identification will return UNKNOWN until enrollment.")
gallery_lock = threading.Lock()

available = ort.get_available_providers()
use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
print("ONNX providers:", available)
print("Using providers:", providers)

face_app = FaceAnalysis(
    name=FACE_MODEL_NAME,
    providers=providers,
    allowed_modules=["detection", "recognition"],
)
face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

def parse_indices() -> list[int]:
    if CAM_INDICES_ENV:
        out = []
        for part in CAM_INDICES_ENV.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                pass
        if out:
            return out
    return [CAM_INDEX, 0, 1, 2]


def open_camera():
    tried = []
    unique = []
    seen = set()
    for idx in parse_indices():
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(idx)

    backends = [("V4L2", cv2.CAP_V4L2), ("DEFAULT", None)]
    for idx in unique:
        for backend_name, backend in backends:
            if backend is None:
                cap_local = cv2.VideoCapture(idx)
            else:
                cap_local = cv2.VideoCapture(idx, backend)
            cap_local.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            ok, _ = cap_local.read()
            if ok:
                return cap_local, idx, backend_name
            tried.append(f"{idx}/{backend_name}")
            cap_local.release()

    raise SystemExit("Could not open camera. Tried: " + ", ".join(tried))


cap, selected_cam, selected_backend = open_camera()
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

    with gallery_lock:
        gallery_items = list(gallery.items())

    if not gallery_items:
        # Service should remain usable even before first enrollment.
        return {"status": "unknown", "name": None, "score": 0.0}

    best_name, best_score = None, -1.0
    for name, mean_emb in gallery_items:
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


def sanitize_name(raw: str) -> str:
    cleaned = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in ("_", "-", " "))
    cleaned = " ".join(cleaned.split())
    return cleaned


def enroll_once(name: str, samples: int):
    person = sanitize_name(name)
    if not person:
        return {"status": "invalid_name", "detail": "Name is empty after sanitization."}

    embs = []
    with cap_lock:
        for _ in range(ENROLL_MAX_FRAMES):
            ret, frame = cap.read()
            if not ret:
                continue

            faces = face_app.get(frame)
            if not faces:
                continue

            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            x1, y1, x2, y2 = map(int, f.bbox)
            area = (x2 - x1) * (y2 - y1)
            if area < 140 * 140:
                continue

            embs.append(f.embedding.astype(np.float32))
            if len(embs) >= samples:
                break

    if len(embs) < samples:
        return {
            "status": "not_enough_samples",
            "name": person,
            "captured": len(embs),
            "required": samples,
        }

    embs_arr = np.stack(embs).astype(np.float32)
    np.savez(os.path.join(DB_DIR, f"{person}.npz"), embs=embs_arr)

    meta = {"person": person, "n_samples": int(len(embs_arr)), "created_unix": time.time()}
    with open(os.path.join(DB_DIR, f"{person}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    norm = embs_arr / np.linalg.norm(embs_arr, axis=1, keepdims=True)
    mean = norm.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    with gallery_lock:
        gallery[person] = mean

    return {
        "status": "enrolled",
        "name": person,
        "samples": int(len(embs_arr)),
        "gallery_size": len(gallery),
    }

@app_web.get("/identify")
def identify():
    return identify_once()

@app_web.post("/enroll")
def enroll(req: EnrollRequest):
    return enroll_once(req.name, req.samples)

@app_web.get("/health")
def health():
    return {"ok": True, "gallery_size": len(gallery), "ts": time.time()}

@app_web.get("/")
def index():
    return {
        "ok": True,
        "service": "face-recognizer",
        "endpoints": ["/identify", "/enroll", "/health"],
    }

@app_web.on_event("shutdown")
def on_shutdown():
    cap.release()

if __name__ == "__main__":
    # Local only, no external access
    print(
        f"Config: model={FACE_MODEL_NAME}, cam={selected_cam}, backend={selected_backend}, "
        f"size={W}x{H}, det_size={DET_SIZE}, flush={FRAME_FLUSH_READS}"
    )
    uvicorn.run(app_web, host="127.0.0.1", port=8008)
