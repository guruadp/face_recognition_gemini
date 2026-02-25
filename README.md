# Face Recognition Project

Realtime face enrollment, recognition, and a local HTTP recognition service (FastAPI), plus a Gemini-based assistant client.

## Project structure

- `scripts/enroll.py`: collect face embeddings for one person and save to `db/<name>.npz` + `db/<name>.json`.
- `scripts/recognize.py`: live webcam recognition window (`KNOWN`, `FAMILIAR`, `UNKNOWN`, `TOO FAR`).
- `scripts/recognizer_service.py`: local API service for `/identify`, `/enroll`, `/reload_db`, `/health`.
- `scripts/gemini_agent.py`: terminal chatbot that calls Gemini and recognition service.
- `scripts/bench_fps.py`: benchmark InsightFace FPS.
- `scripts/list_models.py`: list available Gemini models.
- `db/`: saved face embeddings and metadata.

## Requirements

- Python 3.10+ recommended
- Linux webcam support (`v4l2-ctl` optional but useful)
- CUDA GPU optional (falls back to CPU if CUDA provider is unavailable)

## Setup

```bash
clone this repo
cd into the repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Camera selection

List devices:

```bash
v4l2-ctl --list-devices
```

Set camera via env:

- `CAM_INDEX=2` (single preferred index)
- `CAM_INDICES=2,3,0,1` (fallback order)

Example:

```bash
CAM_INDEX=2 python3 scripts/recognize.py
```

## Enrollment

`enroll.py` now requires the person name:

```bash
python3 scripts/enroll.py <name>
```

Example:

```bash
CAM_INDEX=2 MIN_FACE_AREA=4900 python3 scripts/enroll.py guru
```

Notes:

- Default samples: `20` (`N_SAMPLES` inside script).
- Output files: `db/guru.npz` and `db/guru.json`.
- `MIN_FACE_AREA` controls "move closer" strictness (`70*70` default in `enroll.py`).

## Live recognition UI

```bash
python3 scripts/recognize.py
```

Example tuned for farther faces:

```bash
CAM_INDEX=2 DET_EDGE=640 MIN_FACE_AREA=2500 python3 scripts/recognize.py
```

Useful env vars for `recognize.py`:

- `CAM_INDEX` (default `2`)
- `CAM_INDICES` (fallback order)
- `CAM_WIDTH` / `CAM_HEIGHT` (default `640x480`)
- `DET_EDGE` (default `320`)
- `MIN_FACE_AREA` (default `50*50`)
- `FACE_MODEL_NAME` (default `buffalo_l`)

## Recognition API service

Start service:

```bash
python3 scripts/recognizer_service.py
```

Runs on `http://127.0.0.1:8008`.

Service env vars:

- `CAM_INDEX` (default `0`)
- `CAM_INDICES`
- `CAM_WIDTH` / `CAM_HEIGHT`
- `DET_EDGE`
- `FACE_MODEL_NAME`
- `FRAME_FLUSH_READS`

Health check:

```bash
curl http://127.0.0.1:8008/health
```

Identify once:

```bash
curl http://127.0.0.1:8008/identify
```

Enroll via API:

```bash
curl -X POST http://127.0.0.1:8008/enroll \
  -H "Content-Type: application/json" \
  -d '{"name":"guru","samples":20}'
```

Reload DB without restart:

```bash
curl -X POST http://127.0.0.1:8008/reload_db
```

## Gemini assistant client

Set API key in environment:

```bash
export GEMINI_API_KEY="your_key_here"
```

Or create `scripts/.env`:

```bash
GEMINI_API_KEY=your_key_here
```

Run:

```bash
python3 scripts/gemini_agent.py
```

The agent calls:

- `GET /identify` when user asks identity-style questions.
- `POST /enroll` when user agrees to save face.

Optional env vars:

- `RECOGNIZER_BASE_URL` (default `http://127.0.0.1:8008`)
- `GEMINI_MODEL` (default `gemini-2.5-flash`)
- `ENROLL_SAMPLES` (default `20`)

## Utility scripts

Benchmark FPS:

```bash
python3 scripts/bench_fps.py
```

Example:

```bash
CAM_INDEX=2 DET_EDGE=640 python3 scripts/bench_fps.py
```

List Gemini models:

```bash
export GEMINI_API_KEY="your_key_here"
python3 scripts/list_models.py
```

## Common troubleshooting

- `ModuleNotFoundError: insightface`: activate venv and reinstall deps.
- Camera open failure: set `CAM_INDEX`/`CAM_INDICES` based on `v4l2-ctl --list-devices`.
- Too many `TOO FAR` results: lower `MIN_FACE_AREA` and/or increase `DET_EDGE`.
- Slow performance: confirm CUDA provider appears in startup logs.

