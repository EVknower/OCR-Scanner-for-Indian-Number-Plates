import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_lost=10):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._lost_count = {}

    def update(self, detections):
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = det["bbox"]
                self._lost_count[self.next_id] = 0
                det["track_id"] = self.next_id
                self.next_id += 1
            return detections

        matched_track_ids = set()
        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, tbbox in self.tracks.items():
                score = iou(det["bbox"], tbbox)
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None:
                det["track_id"] = best_id
                self.tracks[best_id] = det["bbox"]
                self._lost_count[best_id] = 0
                matched_track_ids.add(best_id)
            else:
                det["track_id"] = self.next_id
                self.tracks[self.next_id] = det["bbox"]
                self._lost_count[self.next_id] = 0
                matched_track_ids.add(self.next_id)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self._lost_count[tid] += 1
                if self._lost_count[tid] > self.max_lost:
                    del self.tracks[tid]
                    del self._lost_count[tid]

        return detections
