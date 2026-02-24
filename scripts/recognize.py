import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
import sys

DB_DIR = "db"
CAM_INDEX = 0
W, H = 1280, 720
DET_SIZE = (640, 640)

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
    raise SystemExit("DB empty. Run enroll.py first.")

print("Python:", sys.executable)
print("onnxruntime:", ort.__version__)
print("onnxruntime file:", ort.__file__)
available = ort.get_available_providers()
print("ONNX available providers:", available)

use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
if not use_cuda:
    print("\n[WARN] CUDAExecutionProvider NOT available. Recognition will run on CPU.\n")

app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

last_greet_t = {}

print("[recognize] ESC to quit.")

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
        for name, mean_emb in gallery.items():
            s = cos_sim(emb, mean_emb)
            if s > best_score:
                best_score = s
                best_name = name

        if best_score >= THRESH_NAME:
            status = f"KNOWN: {best_name} ({best_score:.2f})"
            color = (0, 255, 0)
            greet_key = f"known:{best_name}"
            if (now - last_greet_t.get(greet_key, 0.0)) > COOLDOWN_SEC:
                print(f"[greet] Hello {best_name} (score={best_score:.2f})")
                last_greet_t[greet_key] = now
            recognized_count += 1
        elif best_score >= THRESH_FAMILIAR:
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
