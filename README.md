# ANPR System — Real-Time Number Plate Recognition

## Stack
- **Detection**: YOLOv8n (pretrained on license plates)
- **Tracking**: IoU-based tracker (ByteTrack-style)
- **OCR**: PaddleOCR
- **DB**: SQLite
- **Export**: Pandas + OpenPyXL
- **Dashboard**: Streamlit

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download pretrained YOLO model
python download_model.py

# 3. Run headless (OpenCV window)
python main.py

# 4. Run dashboard
streamlit run dashboard.py
```

---

## Config (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index or RTSP URL |
| `PROCESS_EVERY_N_FRAMES` | `2` | Skip frames for speed |
| `FRAMES_TO_COLLECT` | `20` | Frames per track before OCR |
| `COOLDOWN_SECONDS` | `300` | Duplicate suppression window |
| `YOLO_CONF_THRESHOLD` | `0.5` | Min detection confidence |

To use IP camera: set `CAMERA_SOURCE = "rtsp://your-ip/stream"`

---

## Folder Structure

```
ANPR-System/
├── models/           ← YOLO .pt file (auto-downloaded)
├── database/         ← vehicles.db (SQLite)
├── exports/          ← Excel reports
├── snapshots/        ← Per-detection vehicle images
├── detection/
│   ├── detector.py   ← YOLOv8 plate detector
│   ├── tracker.py    ← IoU tracker
│   └── quality.py    ← Sharpness/brightness filter
├── ocr/
│   └── paddle_reader.py  ← OCR + multi-frame voting
├── config.py
├── validator.py      ← Indian plate regex
├── database_manager.py
├── export_excel.py
├── dashboard.py      ← Streamlit UI
├── download_model.py
└── main.py           ← Entry point
```

---

## Notes
- First run downloads the YOLO model (~6MB)
- PaddleOCR downloads its own models on first run (~100MB)
- Snapshots saved to `snapshots/` with plate number + timestamp in filename
- Duplicate suppression: same plate won't log again for 5 minutes
