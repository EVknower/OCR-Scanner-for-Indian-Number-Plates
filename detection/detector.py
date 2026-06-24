from ultralytics import YOLO
from config import MODEL_PATH, YOLO_CONF_THRESHOLD
from download_model import download_model

class PlateDetector:
    def __init__(self):
        download_model()
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            crop = frame[y1:y2, x1:x2]
            plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "crop": crop
            })
        return plates
