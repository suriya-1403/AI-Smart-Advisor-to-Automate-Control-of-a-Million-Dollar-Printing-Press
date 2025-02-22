import cv2
import pytesseract
from PIL import Image
import numpy as np

def find_text_roi_refined(thresh, x, y, w, h, max_expansion=350, text_threshold=10):
    """
    Enhances text detection above the checkbox by increasing sensitivity and ensuring proper ROI expansion.

    :param thresh: Thresholded binary image (inverted: text is white, background is black)
    :param x: Initial x-coordinate of the checkbox
    :param y: Initial y-coordinate of the checkbox
    :param w: Width of the checkbox
    :param h: Height of the checkbox
    :param max_expansion: Maximum expansion allowed
    :param text_threshold: The minimum number of text pixels required to stop upward expansion
    :return: Refined ROI coordinates (x, y, width, height)
    """
    height, width = thresh.shape

    # Initialize best bounding box
    best_x, best_y, best_w, best_h = x, y, w, h
    max_text_pixels = 0  # Track the highest detected text pixels

    # First, check above the checkbox, stopping at the first significant text line
    y_above_start = y
    for i in range(y, max(0, y - max_expansion), -1):  # Move upwards
        above_text_pixels = np.count_nonzero(thresh[i, x:x + w])
        if above_text_pixels > text_threshold:
            y_above_start = i  # Stop here since text is detected
            break

    if y_above_start < y:
        best_y = y_above_start
        best_h = y - y_above_start

    else:
        # If no text is found above, proceed with right/left expansion
        directions = ["right", "left"]
        direction_bounds = {
            "right": (x + w, min(x + w + max_expansion, width), 1, y, y + h),
            "left": (x, max(x - max_expansion, 0), -1, y, y + h),
        }

        for direction in directions:
            start, end, step, fixed_start, fixed_end = direction_bounds[direction]
            expansion_gap = 0
            text_pixels = 0
            new_x, new_w = x, w

            for i in range(start, end, step):
                segment = thresh[fixed_start:fixed_end, i]
                nonzero_count = cv2.countNonZero(segment)

                if nonzero_count > 0:
                    text_pixels += nonzero_count
                    expansion_gap = 0  # Reset gap counter

                    if direction == "right":
                        new_w = i - x
                    elif direction == "left":
                        new_x = i
                        new_w = x + w - i

                else:
                    expansion_gap += 1
                    if expansion_gap > 50:  # Stop if too many blank pixels
                        break

            if text_pixels > max_text_pixels:
                max_text_pixels = text_pixels
                best_x, best_w = new_x, new_w

    return best_x, best_y, best_w, best_h


# Load the image again for processing
image = cv2.imread("page0.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours to detect checkboxes
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check if the contour is a rectangle
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Filter by aspect ratio and size (likely a checkbox)
        if 0.8 < aspect_ratio < 1.2 and 10 < w < 50 and 10 < h < 50:
            # Find the text ROI
            roi_x, roi_y, roi_w, roi_h = find_text_roi_refined(thresh, x, y, w, h)

            # Extract and save the text ROI
            text_roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            cv2.imwrite(f"text_roi_{x}_{y}.jpg", text_roi)

            # Draw ROI on the image for visualization
            cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

# Save the updated annotated image
final_annotated_image_path = "final_optimized_annotated_image.jpg"
cv2.imwrite(final_annotated_image_path, image)

# Provide the user with the new processed image
final_annotated_image_path

