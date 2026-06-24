# ANPR System — Real-Time Number Plate Recognition

## Stack
- **Detection**: YOLOv8n (pretrained on license plates)
- **Tracking**: IoU-based tracker (ByteTrack-style)
- **OCR**: PaddleOCR (with mirrored plate support)
- **DB**: SQLite
- **Export**: CSV via Pandas
- **Dashboard**: Streamlit

---

## Quick Start

### Step 1 — Create a Virtual Environment

A virtual environment keeps all packages isolated and prevents global conflicts.

**VS Code / Terminal (Windows):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**VS Code / Terminal (macOS/Linux):**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ You'll know it's active when you see `(venv)` at the start of your terminal prompt.

---

### Step 2 — Select the Virtual Environment in Your IDE

#### VS Code
1. Press `Ctrl + Shift + P` → type **"Python: Select Interpreter"**
2. Choose the one that says `.\venv\Scripts\python.exe` (Windows) or `./venv/bin/python` (macOS/Linux)
3. Open a **New Terminal** in VS Code — it will automatically activate the venv

#### PyCharm
1. Go to **File → Settings → Project → Python Interpreter**
2. Click the gear icon → **Add Interpreter → Add Local Interpreter**
3. Select **Existing environment** and point to `venv/Scripts/python.exe`

#### Antigravity IDE
1. Open the integrated terminal
2. Run `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
3. Run `streamlit run dashboard.py` to start the web app, or `python main.py` for the OpenCV window
4. *(Optional)* You can also just ask the AI agent: *"Run the Streamlit dashboard"* and it will execute it for you!

#### Any Other IDE
- Just make sure the terminal is using the venv Python, not the system Python
- Run `python --version` and check the path matches your `venv/` folder

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Do not install globally.** Always activate the venv first (Step 1) before running this.

---

### Step 4 — Download the YOLO Model

```bash
python download_model.py
```

This downloads the pretrained license plate detection model (~6 MB) into the `models/` folder. Only needed once.

---

### Step 5 — Run the Application

**Option A: Streamlit Dashboard (Recommended)**
Provides a modern web UI with live camera feed, detection log, search, and CSV export.
```bash
streamlit run dashboard.py
```
Then open `http://localhost:8501` in your browser.

**Option B: Headless OpenCV Window**
Runs a standalone OpenCV window. Press `Q` to quit.
```bash
python main.py
```

---

## Changing the Camera Source

Edit **line 4** of [`config.py`](config.py):

```python
CAMERA_SOURCE = 0                    # Built-in webcam (default)
CAMERA_SOURCE = 1                    # Second USB webcam
CAMERA_SOURCE = "rtsp://user:pass@192.168.1.100:554/stream"  # IP/CCTV camera
CAMERA_SOURCE = "http://192.168.1.105:8080/video"            # Phone (IP Webcam app)
CAMERA_SOURCE = r"C:\Videos\test.mp4"                        # Pre-recorded video file
```

---

## Config Reference (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam index or RTSP/HTTP URL |
| `PROCESS_EVERY_N_FRAMES` | `2` | Skip frames for speed |
| `FRAMES_TO_COLLECT` | `3` | Frames per track before OCR runs |
| `COOLDOWN_SECONDS` | `60` | Seconds before the same plate can be re-logged |
| `YOLO_CONF_THRESHOLD` | `0.3` | Min YOLO detection confidence (0–1) |
| `OCR_CONF_THRESHOLD` | `0.34` | Min OCR character confidence (0–1) |
| `SHARPNESS_THRESHOLD` | `20.0` | Min sharpness score to accept a frame |

---

## Folder Structure

```
ANPR-System/
├── models/               ← YOLO .pt file (auto-downloaded)
├── database/             ← vehicles.db (SQLite, auto-created)
├── exports/              ← CSV reports (generated on export)
├── snapshots/            ← Per-detection images with plate + timestamp
├── detection/
│   ├── detector.py       ← YOLOv8 plate detector
│   ├── tracker.py        ← IoU tracker
│   └── quality.py        ← Sharpness/brightness frame filter
├── ocr/
│   └── paddle_reader.py  ← OCR engine + mirrored plate support + voting
├── config.py             ← All tunable parameters
├── validator.py          ← Indian plate regex validation
├── database_manager.py   ← SQLite read/write with cooldown logic
├── export_excel.py       ← CSV export functions
├── dashboard.py          ← Streamlit web dashboard
├── download_model.py     ← Auto-downloads YOLO weights
└── main.py               ← Core ANPR pipeline (threaded OCR)
```

---

## Common Issues & Fixes

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | venv not activated — run `.\venv\Scripts\activate` first |
| Camera feed freezes | Another app is using the camera — close it and restart |
| No detections / green box only | Low lighting — improve brightness, OCR still runs in background |
| Wrong plate on display | Old result shown — wait for current OCR batch to finish (watch "🔍 Scanning…") |
| `shm.dll` error (Windows) | Broken PyTorch — run: `pip uninstall torch torchvision ; pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Camera index not working | Try `CAMERA_SOURCE = 1` or `2` in `config.py` |

---

## Notes
- First run downloads the YOLO model (~6 MB) and PaddleOCR models (~100 MB) — this is one-time only
- OCR runs in a **background thread** — the camera feed is always smooth
- Both mirrored and non-mirrored plates are automatically handled
- Supports single-row and two-row (bike) Indian number plates
- Snapshots saved to `snapshots/` as `<PLATE>_<TIMESTAMP>.jpg`
