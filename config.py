MODEL_PATH = "models/yolo_plate.pt"
MODEL_URL = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"

CAMERA_SOURCE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PROCESS_EVERY_N_FRAMES = 2

YOLO_CONF_THRESHOLD = 0.3

SHARPNESS_THRESHOLD = 20.0
BRIGHTNESS_MIN = 20
BRIGHTNESS_MAX = 240
FRAMES_TO_COLLECT = 3
TOP_FRAMES = 2

OCR_CONF_THRESHOLD = 0.34

PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}'

DB_PATH = "database/vehicles.db"
SNAPSHOT_DIR = "snapshots"
EXPORT_DIR = "exports"

COOLDOWN_SECONDS = 300
