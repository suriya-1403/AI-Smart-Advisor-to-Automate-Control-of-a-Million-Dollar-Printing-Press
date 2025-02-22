# # import module
# from pdf2imageSample import convert_from_path
#
# # Store Pdf with convert_from_path function
# images = convert_from_path('/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential/PWP Barcelona GEC Prints on Light Media Navigator Offset Premium Offset Uncoated Smooth 80 gsm-54lb Text Jun 22.pdf')
#
# for i in range(len(images)):
#     # Save pages as images in the pdf
#     images[i].save('page' + str(i+1) + '.jpg', 'JPEG')

# import cv2

# Load the image
# image_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/page0.jpg"  # Replace with your JPG file path
import cv2
import pytesseract
from PIL import Image
import numpy as np


# Configure Tesseract executable path (if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re


def expand_roi_based_on_text(thresh, x, y, w, h, direction="right", step=1, max_expand=100, threshold=0.1):
    """
    Dynamically expand the ROI based on text boundaries in the specified direction.

    :param thresh: Thresholded binary image (inverted: text is white, background is black)
    :param x: Initial x-coordinate of the ROI
    :param y: Initial y-coordinate of the ROI
    :param w: Width of the checkbox
    :param h: Height of the checkbox
    :param direction: Direction to expand ('up', 'down', 'left', 'right')
    :param step: Step size for expansion
    :param max_expand: Maximum expansion allowed
    :param threshold: Minimum pixel density to stop expansion
    :return: Expanded ROI coordinates (x, y, width, height)
    """
    height, width = thresh.shape
    new_x, new_y, new_w, new_h = x, y, w, h

    for i in range(step, max_expand + 1, step):
        if direction == "right":
            if new_x + new_w + step >= width:
                break
            roi = thresh[new_y:new_y + new_h, new_x + new_w:new_x + new_w + step]
        elif direction == "left":
            if new_x - step < 0:
                break
            roi = thresh[new_y:new_y + new_h, new_x - step:new_x]
        elif direction == "down":
            if new_y + new_h + step >= height:
                break
            roi = thresh[new_y + new_h:new_y + new_h + step, new_x:new_x + new_w]
        elif direction == "up":
            if new_y - step < 0:
                break
            roi = thresh[new_y - step:new_y, new_x:new_x + new_w]
        else:
            break

        # Check pixel density
        white_pixel_density = cv2.countNonZero(roi) / roi.size
        if white_pixel_density < threshold:
            break

        # Adjust ROI coordinates
        if direction == "right":
            new_w += step
        elif direction == "left":
            new_x -= step
            new_w += step
        elif direction == "down":
            new_h += step
        elif direction == "up":
            new_y -= step
            new_h += step

    return new_x, new_y, new_w, new_h


# Load the original image
image_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/page0.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours to detect checkboxes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare a list to store checkbox information
checkbox_text_map = []

# Iterate through contours
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check if the contour is a rectangle
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Filter by aspect ratio and size
        if 0.8 < aspect_ratio < 1.2 and 10 < w < 50 and 10 < h < 50:
            # Expand ROI dynamically in all directions
            x, y, w, h = expand_roi_based_on_text(thresh, x, y, w, h, "right", step=2)
            x, y, w, h = expand_roi_based_on_text(thresh, x, y, w, h, "left", step=2)
            x, y, w, h = expand_roi_based_on_text(thresh, x, y, w, h, "down", step=2)
            x, y, w, h = expand_roi_based_on_text(thresh, x, y, w, h, "up", step=2)

            # Extract the expanded ROI
            text_roi = thresh[y:y + h, x:x + w]

            if text_roi.size > 0:  # Ensure the ROI is non-empty
                # Convert ROI to PIL format for OCR
                text_roi_pil = Image.fromarray(text_roi)

                # Use Tesseract for text extraction
                extracted_text = pytesseract.image_to_string(text_roi_pil, config='--psm 6').strip()

                # Post-process extracted text
                cleaned_text = re.sub(r'\s+', ' ', extracted_text)
                cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,]', '', cleaned_text)
                if not cleaned_text:
                    cleaned_text = "(No text found)"

                # Determine checkbox status (checked or unchecked)
                checkbox_region = thresh[y:y + h, x:x + w]
                total_pixels = checkbox_region.size
                black_pixels = cv2.countNonZero(checkbox_region)
                is_checked = black_pixels > 0.2 * total_pixels

                checkbox_status = "Checked" if is_checked else "Unchecked"
                color = (0, 255, 0) if is_checked else (0, 0, 255)

                # Draw the checkbox and text on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, checkbox_status, (x + w + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(image, cleaned_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Store the information
                checkbox_text_map.append({
                    "checkbox_coords": (x, y, w, h),
                    "status": checkbox_status,
                    "text": cleaned_text
                })

# Display and save the annotated image
cv2.imshow("Checkbox Detection", image)
cv2.imwrite("checkbox_with_text_refined.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the extracted information
print("\nExtracted Information:")
for item in checkbox_text_map:
    print(f"Checkbox at {item['checkbox_coords']} is {item['status']} with text: '{item['text']}'")

# Path to the Tesseract executable (required for Windows users)
# For Linux/MacOS, this step is usually not necessary
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Load the image
image_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/page0.jpg"
image = Image.open(image_path)

# Use Tesseract to extract text
text = pytesseract.image_to_string(image)

# Print the extracted text
print("Extracted Text:")
print(text)
