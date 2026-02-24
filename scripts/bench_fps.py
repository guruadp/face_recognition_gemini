import time
import cv2
from insightface.app import FaceAnalysis
import onnxruntime as ort
import sys
import os
import numpy as np

CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_INDICES_ENV = os.getenv("CAM_INDICES", "").strip()
W = int(os.getenv("CAM_WIDTH", "640"))
H = int(os.getenv("CAM_HEIGHT", "480"))
DET_EDGE = int(os.getenv("DET_EDGE", "320"))
DET_SIZE = (DET_EDGE, DET_EDGE)
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
USE_CAMERA = os.getenv("USE_CAMERA", "1").strip().lower() not in ("0", "false", "no")
N_WARMUP = 30
N_TEST = 200
PROCESS_EVERY_N = 1   # keep 1 for benchmarking; set to 2/3 later in real app

def now():
    return time.perf_counter()

print("Python:", sys.executable)
print("onnxruntime:", ort.__version__)
print("onnxruntime file:", ort.__file__)
available = ort.get_available_providers()
print("ONNX available providers:", available)

use_cuda = "CUDAExecutionProvider" in available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

if not use_cuda:
    print("\n[WARN] CUDAExecutionProvider NOT available. You're benchmarking CPU only.")
    print("Fix: ensure your env has working onnxruntime-gpu + CUDA libs.\n")

# Init InsightFace
app = FaceAnalysis(
    name=FACE_MODEL_NAME,
    providers=providers,
    allowed_modules=["detection", "recognition"],
)

# NOTE: ctx_id isn't the switch for ORT; providers are.
# Still, InsightFace expects ctx_id; use 0 when cuda available else -1.
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
                print(f"Camera opened: index={idx}, backend={backend_name}")
                return cap_local, idx, backend_name
            tried.append(f"{idx}/{backend_name}")
            cap_local.release()

    raise RuntimeError("Camera read failed for all candidates: " + ", ".join(tried))

cap = None
source_label = "synthetic"
if USE_CAMERA:
    try:
        cap, selected_cam, selected_backend = open_camera()
        source_label = f"camera(index={selected_cam}, backend={selected_backend})"
    except Exception as e:
        print(f"[WARN] {e}")
        print("[WARN] Falling back to synthetic frames. Set USE_CAMERA=0 to force this mode.")

print(
    f"Config: model={FACE_MODEL_NAME}, source={source_label}, "
    f"size={W}x{H}, det_size={DET_SIZE}"
)
synthetic_frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

def grab():
    if cap is None:
        return synthetic_frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    return frame

# Warmup
for _ in range(N_WARMUP):
    frame = grab()
    _ = app.get(frame)

# Benchmark
t0 = now()
faces_total = 0
processed_frames = 0

for i in range(N_TEST):
    frame = grab()

    # For a realistic app you might skip frames; keep for benchmarking
    if (i % PROCESS_EVERY_N) != 0:
        continue

    faces = app.get(frame)
    faces_total += len(faces)
    processed_frames += 1

t1 = now()
if cap is not None:
    cap.release()

dt = t1 - t0
fps = processed_frames / dt
print(f"\nProcessed {processed_frames} frames in {dt:.2f}s => {fps:.1f} FPS")
print(f"Avg faces/frame: {faces_total / max(processed_frames,1):.2f}")

print("\nTip: In another terminal run:  watch -n 0.5 nvidia-smi")
print("If GPU util stays ~0%, you are not actually using CUDA.\n")
