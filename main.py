import cv2
import os
import threading
import queue
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

        # OCR runs in a background thread so it never blocks the camera loop
        self._ocr_queue = queue.Queue(maxsize=4)
        self._last_detections = []          # shared result, written by OCR thread
        self._lock = threading.Lock()
        self._ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self._ocr_thread.start()

        # Tracks currently queued for OCR (avoid re-queuing while one is running)
        self._queued_tids = set()

    # ------------------------------------------------------------------ #
    #  Background OCR worker                                               #
    # ------------------------------------------------------------------ #
    def _ocr_worker(self):
        while True:
            try:
                tid, frames, snap_frame, bbox = self._ocr_queue.get(timeout=1)
            except queue.Empty:
                continue

            print(f"[ANPR] Running OCR on track_id={tid} ({len(frames)} frames)")
            plate_text, confidence = vote(frames)
            print(f"[ANPR] OCR result: '{plate_text}' conf={confidence:.2f}")

            with self._lock:
                self._queued_tids.discard(tid)

            if plate_text and is_valid(plate_text):
                plate_text = clean(plate_text)
                print(f"[ANPR] Valid plate: '{plate_text}'")
                snap_path = self.save_snapshot(snap_frame, plate_text)
                db.insert(plate_text, confidence, snap_path)
                # Always update the display, even if it's a duplicate in the DB
                print(f"[ANPR] ✅ {plate_text} ({confidence*100:.1f}%)")
                with self._lock:
                    self._last_detections.append({"plate": plate_text, "confidence": confidence})
            else:
                print(f"[ANPR] ❌ Invalid: '{plate_text}'")

            self._ocr_queue.task_done()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def save_snapshot(self, frame, plate):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plate}_{ts}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, frame)
        return path

    def flush_detections(self):
        """Pop and return any new detections since last call."""
        with self._lock:
            out = list(self._last_detections)
            self._last_detections.clear()
        return out

    # ------------------------------------------------------------------ #
    #  Main per-frame method (NEVER blocks on OCR)                        #
    # ------------------------------------------------------------------ #
    def process_frame(self, frame, frame_count):
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            return frame

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

            # Queue OCR when enough frames are collected, but only if not already queued
            buf = self.frame_buffer[tid]
            with self._lock:
                already_queued = tid in self._queued_tids
            if len(buf) >= FRAMES_TO_COLLECT and not already_queued:
                frames_copy = list(buf)
                self.frame_buffer[tid] = []
                with self._lock:
                    self._queued_tids.add(tid)
                try:
                    self._ocr_queue.put_nowait(
                        (tid, frames_copy, frame.copy(), det["bbox"])
                    )
                except queue.Full:
                    # Drop this batch if worker is busy — camera smoothness wins
                    with self._lock:
                        self._queued_tids.discard(tid)

        return frame


# ------------------------------------------------------------------ #
#  Standalone OpenCV runner                                            #
# ------------------------------------------------------------------ #
def run():
    pipeline = ANPRPipeline()
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[ANPR] Starting... Press Q to quit.")
    frame_count = 0
    last_plate_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ANPR] No frame received. Check camera source.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        annotated = pipeline.process_frame(frame, frame_count)

        # Collect any results the background OCR thread finished
        for det in pipeline.flush_detections():
            last_plate_text = f"{det['plate']} ({det['confidence']*100:.1f}%)"

        if last_plate_text:
            cv2.putText(annotated, last_plate_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        cv2.imshow("ANPR - Press Q to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
