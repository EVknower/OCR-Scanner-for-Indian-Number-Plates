# ANPR System Architecture and Detailed Explanation

This document provides a comprehensive breakdown of the Automatic Number Plate Recognition (ANPR) system, detailing the models used, the project structure, and the step-by-step data flow.

---

## 1. The Machine Learning Models

The system relies on two main deep learning models to achieve real-time performance and high accuracy:

### A. Plate Detection Model: YOLOv8 (You Only Look Once, version 8)
* **What it does:** Scans the entire video frame and finds the exact bounding box (coordinates) where a license plate is located.
* **Why YOLOv8:** It is extremely fast and designed for real-time object detection. We use the "nano" version (`YOLOv8n`), which is optimized for speed on CPUs.
* **Where it lives:** Downloaded automatically into the `models/yolo_plate.pt` file.
* **Code location:** `detection/detector.py`

### B. Optical Character Recognition (OCR) Model: PaddleOCR
* **What it does:** Takes the cropped image of the license plate (provided by YOLO) and extracts the actual text (characters and numbers) from it.
* **Why PaddleOCR:** It is currently one of the most accurate open-source OCR engines, especially good at handling difficult lighting, varying fonts, and multiple lines of text (like on two-row motorcycle plates).
* **Where it lives:** PaddleOCR automatically downloads its own models to your home directory (`~/.paddleocr/`) the first time it runs.
* **Code location:** `ocr/paddle_reader.py`

---

## 2. Step-by-Step Data Flow (How it works)

When you start the system, a background process (`main.py` -> `ANPRPipeline`) begins pulling frames from the camera. Here is the exact lifecycle of a single vehicle passing by:

1. **Frame Capture:** A frame is read from the camera (Webcam, IP Cam, or Video file).
2. **Detection (YOLO):** `PlateDetector` scans the frame. If it finds a plate, it crops that exact portion of the image.
3. **Tracking (IoU Tracker):** The `SimpleTracker` assigns a unique ID (e.g., `ID: 5`) to the plate so the system knows it's the *same* car across multiple frames.
4. **Quality Check:** The `quality.py` script checks the cropped plate image. Is it too blurry? Is it too dark? If it's garbage, it throws it away to save processing time.
5. **Frame Collection:** The system collects the best `3` frames (configurable) of that specific car's plate.
6. **OCR Processing (Background Thread):**
   * The 3 frames are sent to a background queue so the main camera feed doesn't freeze.
   * `paddle_reader.py` applies image enhancements (CLAHE) to make the text pop.
   * It also creates a mirrored (flipped) version of the plate to handle cases where phone screens or front-facing cameras flip the image.
   * PaddleOCR reads the text from all variations.
7. **Voting System:** Since we have 3 frames, the OCR might read: `HR98AA7777`, `HR98AA7777`, and `HR98AA777`. The voting system picks the most common result to ensure maximum accuracy.
8. **Validation:** The winning text is passed to `validator.py`. A Regular Expression (Regex) checks if the text perfectly matches the Indian Number Plate format (e.g., 2 letters, 1-2 numbers, 1-3 letters, 4 numbers).
9. **Storage:** If valid, the plate number, confidence score, and a snapshot image are saved to the SQLite database (`database/vehicles.db`) via `database_manager.py`. A cooldown prevents logging the same car again for 60 seconds.
10. **Dashboard Update:** The Streamlit dashboard (`dashboard.py`) instantly reads the new entry and displays it on the screen.

---

## 3. Project File Structure Explained

Here is what every file in the system is responsible for:

### ⚙️ Core Pipeline
* `main.py` -> The heart of the system. It glues everything together, manages the background threads for OCR, and handles the OpenCV rendering.
* `config.py` -> The control panel. You change camera sources, thresholds, and timers here.

### 🔍 Detection Module (`detection/` folder)
* `detector.py` -> Loads the YOLO model and finds the plates in the frame.
* `tracker.py` -> Assigns IDs to moving plates so we track cars over time.
* `quality.py` -> Calculates image sharpness and brightness.

### 📝 OCR Module (`ocr/` folder)
* `paddle_reader.py` -> Loads PaddleOCR, enhances the image (CLAHE), handles mirrored images, sorts bounding boxes (for two-row plates), and runs the voting logic.

### 🛡️ Validation & Data
* `validator.py` -> Contains the Indian License Plate Regex logic to filter out garbage text.
* `database_manager.py` -> Handles saving to and reading from the `vehicles.db` SQLite database. It also enforces the "cooldown" so you don't get 100 entries for a car waiting at a red light.
* `export_excel.py` -> Converts the database records into a CSV file for download.

### 🖥️ User Interface
* `dashboard.py` -> The Streamlit web application. It runs the `main.py` pipeline in the background and renders the live feed, metrics, and search tables in your browser.

### 🛠️ Utilities
* `download_model.py` -> A helper script to fetch the YOLO model from the internet automatically.
* `.gitignore` -> Tells Git to ignore heavy model files and databases so your GitHub repository stays clean.
* `requirements.txt` -> The list of Python packages needed to make the code run.
