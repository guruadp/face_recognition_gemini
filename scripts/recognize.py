import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "db"))
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_INDICES_ENV = os.getenv("CAM_INDICES", "").strip()
W = int(os.getenv("CAM_WIDTH", "640"))
H = int(os.getenv("CAM_HEIGHT", "480"))
DET_EDGE = int(os.getenv("DET_EDGE", "320"))
DET_SIZE = (DET_EDGE, DET_EDGE)
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")

# Start conservative. You will tune these.
THRESH_NAME = 0.50
THRESH_FAMILIAR = 0.40

COOLDOWN_SEC = 8

def load_db(db_dir):
    gallery = {}  # name -> mean embedding (normalized)
    for fn in os.listdir(db_dir):
        if not fn.endswith(".npz"):
            continue
        name = fn[:-4]
        data = np.load(os.path.join(db_dir, fn))
        embs = data["embs"].astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        gallery[name] = mean_emb
    return gallery

def cos_sim(a, b):
    return float(np.dot(a, b))

gallery = load_db(DB_DIR)
if not gallery:
    print("[WARN] DB is empty. Faces will be labeled UNKNOWN until enrollment.")

print("Python:", sys.executable)
print("onnxruntime:", ort.__version__)
print("onnxruntime file:", ort.__file__)
available = ort.get_available_providers()
print("ONNX available providers:", available)

use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
if not use_cuda:
    print("\n[WARN] CUDAExecutionProvider NOT available. Recognition will run on CPU.\n")

app = FaceAnalysis(
    name=FACE_MODEL_NAME,
    providers=providers,
    allowed_modules=["detection", "recognition"],
)
app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

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

last_greet_t = {}

print("[recognize] ESC to quit.")
print(
    f"[recognize] Config: model={FACE_MODEL_NAME}, cam={selected_cam}, "
    f"backend={selected_backend}, size={W}x{H}, det_size={DET_SIZE}"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    now = cv2.getTickCount() / cv2.getTickFrequency()
    recognized_count = 0

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        area = (x2 - x1) * (y2 - y1)

        if area <= 140 * 140:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, "TOO FAR", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            continue

        emb = f.embedding.astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        best_name, best_score = "unknown", -1.0
        if gallery:
            for name, mean_emb in gallery.items():
                s = cos_sim(emb, mean_emb)
                if s > best_score:
                    best_score = s
                    best_name = name

        if gallery and best_score >= THRESH_NAME:
            status = f"KNOWN: {best_name} ({best_score:.2f})"
            color = (0, 255, 0)
            greet_key = f"known:{best_name}"
            if (now - last_greet_t.get(greet_key, 0.0)) > COOLDOWN_SEC:
                print(f"[greet] Hello {best_name} (score={best_score:.2f})")
                last_greet_t[greet_key] = now
            recognized_count += 1
        elif gallery and best_score >= THRESH_FAMILIAR:
            status = f"FAMILIAR ({best_score:.2f})"
            color = (0, 255, 255)
            greet_key = "familiar"
            if (now - last_greet_t.get(greet_key, 0.0)) > COOLDOWN_SEC:
                print(f"[greet] Welcome back (score={best_score:.2f})")
                last_greet_t[greet_key] = now
            recognized_count += 1
        else:
            status = f"UNKNOWN ({best_score:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    top_text = f"Faces: {len(faces)}  Recognized: {recognized_count}"
    cv2.putText(frame, top_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Recognize", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
