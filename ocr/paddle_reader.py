import cv2
import numpy as np
from collections import Counter
from paddleocr import PaddleOCR
from config import OCR_CONF_THRESHOLD, TOP_FRAMES
from detection.quality import score as quality_score

_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return _ocr


def preprocess_variants(image):
    """Return multiple preprocessed versions to try."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Upscale
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    variants = []

    # 1. CLAHE enhanced (better for dark/uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))

    # 2. Adaptive threshold (works in bad light)
    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    # 3. Plain grayscale upscaled (sometimes simpler is better)
    variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    return variants


def read_plate(image):
    ocr = get_ocr()

    # Try both the original and horizontally flipped (handles mirrored plates on phone screens)
    images_to_try = [image, cv2.flip(image, 1)]
    best_text = None
    best_conf = 0.0

    for img in images_to_try:
        for proc_bgr in preprocess_variants(img):
            try:
                result = ocr.ocr(proc_bgr, cls=True)
            except Exception as e:
                print(f"[OCR] Error: {e}")
                continue

            if not result or not result[0]:
                continue

            texts = []
            confs = []
            for line in result[0]:
                text = line[1][0].replace(" ", "").upper()
                conf = line[1][1]
                print(f"[OCR] Raw read: '{text}' conf={conf:.2f}")
                if conf >= OCR_CONF_THRESHOLD:
                    texts.append(text)
                    confs.append(conf)

            if texts:
                combined = "".join(texts)
                avg_conf = float(np.mean(confs))
                if avg_conf > best_conf:
                    best_text = combined
                    best_conf = avg_conf

    return best_text, best_conf


def vote(frames):
    sorted_frames = sorted(frames, key=lambda f: quality_score(f), reverse=True)
    top_frames = sorted_frames[:TOP_FRAMES]
    results = []
    for frame in top_frames:
        text, conf = read_plate(frame)
        if text:
            results.append((text, conf))
            print(f"[OCR] Frame result: '{text}' conf={conf:.2f}")

    if not results:
        print("[OCR] No valid readings from any frame")
        return None, 0.0

    # Vote by most common text
    texts = [r[0] for r in results]
    winner, count = Counter(texts).most_common(1)[0]
    avg_conf = float(np.mean([r[1] for r in results if r[0] == winner]))
    print(f"[OCR] Voted winner: '{winner}' ({count}/{len(results)} votes, conf={avg_conf:.2f})")
    return winner, avg_conf
