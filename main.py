import cv2
import os
import numpy as np
from datetime import datetime
from collections import defaultdict

from config import (
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT,
    PROCESS_EVERY_N_FRAMES, FRAMES_TO_COLLECT, SNAPSHOT_DIR
)
from detection.detector import PlateDetector
from detection.tracker import SimpleTracker
from detection.quality import is_usable
from ocr.paddle_reader import vote
from validator import is_valid, clean
import database_manager as db

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

class ANPRPipeline:
    def __init__(self):
        self.detector = PlateDetector()
        self.tracker = SimpleTracker()
        self.frame_buffer = defaultdict(list)

    def save_snapshot(self, frame, plate):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plate}_{ts}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return path

    def process_frame(self, frame, frame_count):
        detections_out = []

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            return frame, detections_out

        plates = self.detector.detect(frame)
        plates = self.tracker.update(plates)

        for det in plates:
            tid = det.get("track_id", -1)
            crop = det["crop"]

            if crop.size == 0:
                continue

            if is_usable(crop):
                self.frame_buffer[tid].append(crop.copy())

            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(self.frame_buffer[tid]) >= FRAMES_TO_COLLECT:
                print(f"[ANPR] Running OCR on track_id={tid} ({len(self.frame_buffer[tid])} frames)")
                plate_text, confidence = vote(self.frame_buffer[tid])
                self.frame_buffer[tid] = []

                print(f"[ANPR] OCR result: '{plate_text}' conf={confidence:.2f}")
                if plate_text and is_valid(plate_text):
                    plate_text = clean(plate_text)
                    print(f"[ANPR] Valid plate: '{plate_text}'")
                    snap_path = self.save_snapshot(frame, plate_text)
                    inserted = db.insert(plate_text, confidence, snap_path)
                    if inserted:
                        print(f"[ANPR] ✅ {plate_text} ({confidence*100:.1f}%)")
                        detections_out.append({"plate": plate_text, "confidence": confidence})
                    cv2.putText(frame, plate_text, (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                else:
                    print(f"[ANPR] ❌ Invalid/failed validation: '{plate_text}'")

        return frame, detections_out


def run():
    pipeline = ANPRPipeline()
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[ANPR] Starting... Press Q to quit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ANPR] No frame received. Check camera source.")
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        annotated, _ = pipeline.process_frame(frame, frame_count)
        cv2.imshow("ANPR - Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
