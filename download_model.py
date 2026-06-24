import os
import urllib.request
from config import MODEL_PATH, MODEL_URL

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[Model] Found at {MODEL_PATH}")
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[Model] Downloading pretrained plate detector...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[Model] Saved to {MODEL_PATH}")

if __name__ == "__main__":
    download_model()
