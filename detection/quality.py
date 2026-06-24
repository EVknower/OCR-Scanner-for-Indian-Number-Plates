import cv2
import numpy as np
from config import SHARPNESS_THRESHOLD, BRIGHTNESS_MIN, BRIGHTNESS_MAX

def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return np.mean(gray)

def is_usable(image):
    b = brightness(image)
    s = sharpness(image)
    ok = s >= SHARPNESS_THRESHOLD and BRIGHTNESS_MIN <= b <= BRIGHTNESS_MAX
    if not ok:
        print(f"[Quality] REJECTED  sharpness={s:.1f} (min {SHARPNESS_THRESHOLD})  brightness={b:.1f} ({BRIGHTNESS_MIN}-{BRIGHTNESS_MAX})")
    return ok

def score(image):
    return sharpness(image)
