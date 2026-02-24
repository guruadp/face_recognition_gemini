import time
import cv2
from insightface.app import FaceAnalysis
import onnxruntime as ort
import sys

CAM_INDEX = 0
W, H = 1280, 720
DET_SIZE = (640, 640)
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
app = FaceAnalysis(name="buffalo_l", providers=providers)

# NOTE: ctx_id isn't the switch for ORT; providers are.
# Still, InsightFace expects ctx_id; use 0 when cuda available else -1.
app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

def grab():
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
cap.release()

dt = t1 - t0
fps = processed_frames / dt
print(f"\nProcessed {processed_frames} frames in {dt:.2f}s => {fps:.1f} FPS")
print(f"Avg faces/frame: {faces_total / max(processed_frames,1):.2f}")

print("\nTip: In another terminal run:  watch -n 0.5 nvidia-smi")
print("If GPU util stays ~0%, you are not actually using CUDA.\n")