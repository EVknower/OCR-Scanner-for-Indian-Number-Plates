import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

img = cv2.imread("debug_crop_0.jpg")
if img is None:
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "HR98AA7777", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

result = ocr.ocr(img, cls=True)
print("SUCCESS!")
print(f"Result type: {type(result)}")
if result and result[0]:
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        print(f"  Text: '{text}' Conf: {conf:.3f}")
else:
    print("  No text detected")
