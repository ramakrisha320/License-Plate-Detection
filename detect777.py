import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract

# Load Image
img = cv2.imread("plate.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Histogram Equalization
gray = cv2.equalizeHist(gray)

# Apply Bilateral Filtering (Noise Reduction)
filtered = cv2.bilateralFilter(gray, 9, 75, 75)

# Apply Adaptive Thresholding
threshold = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply Canny Edge Detection
edges = cv2.Canny(threshold, 100, 200)

# Use Morphological Operations to refine plate detection
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours to detect possible license plates
contours = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for c in contours:
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) == 4:  # Assuming license plates are roughly rectangular
        x, y, w, h = cv2.boundingRect(approx)
        plate = img[y:y+h, x:x+w]

        # Convert detected region to grayscale for OCR
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.GaussianBlur(plate_gray, (5,5), 0)

        # OCR using EasyOCR
        reader = easyocr.Reader(["en"])
        detection = reader.readtext(plate_gray)
        
        if detection:
            license_plate_text = detection[0][1]
            print(f"Detected License Plate: {license_plate_text}")

            # Draw bounding box around detected plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, license_plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display final processed image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("License Plate Detection")
plt.axis("off")
plt.show()