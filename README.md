# NeuralGuard — Deepfake Detection AI

An end-to-end deepfake detection web app that analyses uploaded videos (and images) frame-by-frame using an **EfficientNet-B3** model **fine-tuned on the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset**, then renders a forensic dashboard with an overall authenticity score, a per-frame timeline heatmap, suspicious-frame thumbnails, and live processing logs.

```
┌────────────────────┐     HTTP/JSON      ┌──────────────────────┐     PyTorch     ┌──────────────────────┐
│  React + Vite UI   │ ─────────────────► │  FastAPI Backend     │ ──────────────► │  EfficientNet-B3     │
│  (Dashboard)       │ ◄───────────────── │  /predict, /predict/ │ ◄────────────── │  Temporal Aggregator │
└────────────────────┘    Heatmap JSON    │  video, /health      │   Scores 0–1    └──────────────────────┘
                                          └──────────────────────┘
```

---

## Features

- **Video deepfake detection** — uniformly samples up to 50 frames per video and produces a per-frame probability score.
- **Image deepfake detection** — single-shot inference on still images.
- **Timeline heatmap** — visualises where in the video the model suspects tampering (green / amber / red).
- **Suspicious frame gallery** — thumbnails of frames whose score exceeds the threshold (`> 0.4`), with timestamps.
- **Live processing log** — terminal-style log panel that streams pipeline state on the dashboard.
- **GPU-aware** — auto-detects CUDA; falls back to CPU.
- **CORS-enabled API** — clean separation between the React frontend and the FastAPI backend.

---

## Tech Stack

| Layer    | Technology                                                          |
| -------- | ------------------------------------------------------------------- |
| Frontend | React 19, Vite 7, plain CSS (Inter font, custom dashboard theming)  |
| Backend  | FastAPI, Uvicorn, python-multipart                                  |
| ML / CV  | PyTorch, torchvision, `timm` (EfficientNet-B3), OpenCV, Pillow, NumPy |
| Model    | EfficientNet-B3 fine-tuned on FaceForensics++, with a custom temporal aggregator |

---

## Project Structure

```
deepfake-app/
├── backend/
│   ├── app.py              # FastAPI entrypoint (/health, /predict, /predict/video)
│   ├── infer.py            # Frame extraction + image/video inference pipeline
│   ├── model.py            # DeepfakeModel (EfficientNet-B3 + temporal mean pool)
│   ├── requirements.txt    # Python dependencies
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # NeuralGuard dashboard
│   │   ├── main.jsx
│   │   ├── index.css       # Theme variables, dashboard layout
│   │   └── App.css
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── model/
│   └── DEEPFAKE_DETECTION_AI.pth   # FaceForensics++ fine-tuned weights (EfficientNet-B3, 1-class head)
├── scratch_inspect_model.py        # Utility to peek inside the .pth checkpoint
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- (Optional) **CUDA-capable GPU** for faster inference
- The model checkpoint `model/DEEPFAKE_DETECTION_AI.pth` (included in this repo)

---

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/Lucifer-cyber007/deepfake-detection-ai.git
cd deepfake-detection-ai
```

### 2) Run the backend (FastAPI)

```bash
cd backend
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

The API now serves on **http://localhost:8000**.

> By default, the backend loads `model/DEEPFAKE_DETECTION_AI.pth` relative to the project root. To override, set the `DEEPFAKE_MODEL_PATH` environment variable to a custom checkpoint path.

### 3) Run the frontend (Vite + React)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually **http://localhost:5173**) and start uploading videos.

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`

Health-check endpoint.

```json
{ "status": "online", "timestamp": 1715500000.123 }
```

### `POST /predict`

Image deepfake detection. Body: `multipart/form-data` with field `file` (any `image/*` MIME type).

```json
{
  "success": true,
  "prediction": {
    "score": 0.87,
    "label": "FAKE",
    "confidence": 0.87
  },
  "filename": "example.jpg",
  "timestamp": 1715500000.456
}
```

### `POST /predict/video`

Video deepfake detection. Body: `multipart/form-data` with field `file` (any `video/*` MIME type).

```json
{
  "success": true,
  "prediction": {
    "overall_score": 0.62,
    "label": "FAKE",
    "confidence": 0.62,
    "heatmap": [
      { "ts": 0.0, "val": 0.41 },
      { "ts": 0.5, "val": 0.78 }
    ],
    "suspicious_frames": [
      { "timestamp": 0.5, "score": 0.78, "thumbnail": "<base64-jpeg>" }
    ]
  },
  "filename": "clip.mp4",
  "timestamp": 1715500000.789
}
```

A score of **> 0.5** is labelled `FAKE`; otherwise `REAL`. The `heatmap` array drives the timeline strip in the UI, and `suspicious_frames` populates the thumbnail gallery (only frames scoring above `0.4` carry a thumbnail).

---

## Model Details

The bundled checkpoint is an **EfficientNet-B3** classifier **fine-tuned on [FaceForensics++](https://github.com/ondyari/FaceForensics)** (FF++), a large-scale benchmark of real and manipulated face videos. The app reuses those learned weights for frame-level scoring on new uploads.

| Property        | Value                                                      |
| --------------- | ---------------------------------------------------------- |
| Training data   | [FaceForensics++](https://github.com/ondyari/FaceForensics) (fine-tuned checkpoint) |
| Backbone        | `efficientnet_b3` (via [`timm`](https://github.com/huggingface/pytorch-image-models)) |
| Pre-training    | ImageNet-style backbone; task-specific weights from FF++ fine-tuning |
| Input size      | 300 × 300                                                  |
| Normalisation   | ImageNet mean / std                                        |
| Head            | `Linear(1536 → 1)` with sigmoid                            |
| Temporal pool   | Mean over `T` frames (clip-level prediction)               |
| Frames sampled  | Up to 50 uniformly spaced frames per video                 |

The model accepts either `(B, C, H, W)` for images or `(B, T, C, H, W)` for video clips and outputs a single logit per sample.

---

## How It Works

1. **Upload** — The user drops a video into the dashboard's dropzone.
2. **Frame sampling** — `infer.extract_frames` opens the video with OpenCV and samples 50 uniformly spaced frames.
3. **Preprocessing** — Each frame is resized to 300×300, ImageNet-normalised, and stacked into a tensor.
4. **Inference** — Frames are fed through EfficientNet-B3; a sigmoid yields per-frame fake probability.
5. **Aggregation** — The overall video score is the mean of per-frame scores. Frames with score `> 0.4` are returned with base64 thumbnails for the gallery.
6. **Dashboard** — React renders the heatmap timeline, suspicious-frame grid, overall risk gauge, and live processing log.

---

## Development Notes

- The hardcoded model path in `infer.py` was replaced with a portable lookup relative to the project root (with `DEEPFAKE_MODEL_PATH` as an override). This means cloning + running works on any machine.
- The frontend currently calls `http://localhost:8000` directly; for production, swap this for an env-based base URL or a Vite proxy.
- The frontend `npm run lint` uses ESLint flat config (`eslint.config.js`).

---

## Roadmap

- [ ] Drag-and-drop visual feedback in the dropzone
- [ ] Server-side persistence of analysis history
- [ ] Face-region cropping (e.g., MTCNN) before inference for stronger signal
- [ ] CSV / ZIP export of analysis artefacts (buttons are present but stubbed)
- [ ] Dockerfile + `docker-compose.yml` for one-command deployment
- [ ] Streaming SSE/WebSocket inference progress to the UI

---

## License

This project is released for academic and research purposes. Add a license file (e.g. MIT) before publishing more widely.

---

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) for the benchmark dataset used to fine-tune the detection model.
- [PyTorch Image Models (`timm`)](https://github.com/huggingface/pytorch-image-models) for the EfficientNet-B3 backbone.
- [FastAPI](https://fastapi.tiangolo.com/) for the snappy backend.
- [Vite](https://vitejs.dev/) + [React](https://react.dev/) for the frontend.
