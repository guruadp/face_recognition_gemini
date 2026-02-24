import os, time, json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
import sys

DB_DIR = "db"
PERSON = "saber"     # change this each time
N_SAMPLES = 20
CAM_INDEX = 0
W, H = 1280, 720
DET_SIZE = (640, 640)

os.makedirs(DB_DIR, exist_ok=True)

print("Python:", sys.executable)
print("onnxruntime:", ort.__version__)
print("onnxruntime file:", ort.__file__)
available = ort.get_available_providers()
print("ONNX available providers:", available)

use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
if not use_cuda:
    print("\n[WARN] CUDAExecutionProvider NOT available. Enrollment will run on CPU.\n")

app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

embs = []
print(f"[enroll] {PERSON}: collect {N_SAMPLES} samples. ESC to exit.")
print("Do: front, slight left/right, slight up/down. Good light. No motion blur.")

while len(embs) < N_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        x1, y1, x2, y2 = map(int, f.bbox)
        area = (x2-x1) * (y2-y1)

        if area > 140 * 140:
            embs.append(f.embedding.copy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {len(embs)}/{N_SAMPLES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Move closer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Enroll", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if not embs:
    raise SystemExit("No samples captured. Fix camera/light/distance.")

embs = np.stack(embs).astype(np.float32)
np.savez(os.path.join(DB_DIR, f"{PERSON}.npz"), embs=embs)

meta = {"person": PERSON, "n_samples": int(len(embs)), "created_unix": time.time()}
with open(os.path.join(DB_DIR, f"{PERSON}.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("[enroll] saved:", os.path.join(DB_DIR, f"{PERSON}.npz"))
