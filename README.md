# License Plate Detection 🚗🔍

## Overview
This project detects license plate text from images using **OpenCV** and **EasyOCR**. The script applies **grayscale conversion, contrast enhancement, edge detection, and text recognition** to extract and overlay detected text on images.

## Features
- **Grayscale conversion** for improved OCR accuracy
- **Histogram equalization** to enhance contrast
- **Edge detection & Morphological operations** to refine license plate extraction
- **OCR-based text recognition** using **EasyOCR**
- **Overlaying detected text** on the processed image

## Installation
To run this project, install the required dependencies:

```bash
pip install opencv-python easyocr numpy matplotlib
Usage
- Place your image file in the project directory (e.g., license_plate.jpg).
- Run the detection script:
python license_plate_detection.py
- The processed image will be displayed with extracted license plate text.
Code Explanation
- Preprocessing Steps:
- Convert image to grayscale
- Apply histogram equalization for better visibility
- Perform noise reduction with bilateral filtering
- Detect edges using Canny edge detection
- Apply morphological transformations for cleaner contours
- OCR Recognition:
- Uses EasyOCR to extract text from the refined plate region
- Visualization:
- Displays processed image with detected text overlay
Future Enhancements
- Improve detection accuracy with deep learning-based OCR models.
- Add support for video frame extraction for real-time detection.
python license_plate_detection.py
