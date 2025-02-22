import cv2
import pytesseract
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("find_text_roi")

def find_text_roi(thresh, x, y, w, h, max_expansion=350):
    """
    Dynamically expands the ROI in all directions, selecting the best one based on detected text.
    Ensures that after moving up, a second pass expands right & left to capture full width.

    :param thresh: Thresholded binary image (inverted: text is white, background is black)
    :param x: Initial x-coordinate of the checkbox
    :param y: Initial y-coordinate of the checkbox
    :param w: Width of the checkbox
    :param h: Height of the checkbox
    :param max_expansion: Maximum number of pixels to expand (prevents infinite expansion)
    :return: Expanded ROI coordinates (x, y, width, height)
    """
    height, width = thresh.shape
    roi_id = f"ROI(x={x}, y={y}, w={w}, h={h})"
    logger.info(f"Starting text ROI detection for {roi_id}")

    # Initialize expansion variables
    best_x, best_y, best_w, best_h = x, y, w, h
    best_direction = None
    max_text_pixels = 0  # Track highest text detection

    # Define search directions
    directions = ["right", "left", "down", "up"]
    direction_bounds = {
        "right": (x + w, min(x + w + max_expansion, width), 1, y, y + h),
        "left": (x, max(x - max_expansion, 0), -1, y, y + h),
        "down": (y + h, min(y + h + max_expansion, height), 1, x, x + w),
        "up": (y, max(y - max_expansion, 0), -3, x, x + w),
    }

    for direction in directions:
        start, end, step, fixed_start, fixed_end = direction_bounds[direction]
        expansion_gap = 0
        text_pixels = 0
        new_x, new_y, new_w, new_h = x, y, w, h

        for i in range(start, end, step):
            if direction in ["right", "left"]:
                segment = thresh[fixed_start:fixed_end, i]
            else:  # "up" or "down"
                segment = thresh[i, fixed_start:fixed_end]

            nonzero_count = cv2.countNonZero(segment)

            if nonzero_count > 0:
                text_pixels += nonzero_count
                expansion_gap = 0  # Reset gap counter

                if direction == "right":
                    new_w = i - x
                elif direction == "left":
                    new_x = i
                    new_w = x + w - i
                elif direction == "down":
                    new_h = i - y
                elif direction == "up":
                    if best_direction == "up":  # If already moving up, keep expanding
                        new_y = i - 10  # Continue moving past the first detected text block
                    else:
                        new_y = i # First detection of text

                    new_h = (y + h) - new_y
                    logger.info("Detected text at y=%d, updating ROI to (x=%d, y=%d, w=%d, h=%d)",
                                i, new_x, new_y, new_w, new_h)

                    # Stop upward movement since we detected text
                    logger.info("Stopping UP expansion at y=%d", i)
                    # Stop upward movement since we detected text
                    break

            else:
                expansion_gap += 1

            # Stop expansion if too many blank pixels are encountered
            if expansion_gap > 50:
                logger.info("Too many blank pixels encountered. Stopping expansion in direction: %s", direction)
                break

        # Update best expansion if more text is detected
        if text_pixels > max_text_pixels or (direction == "up" and best_direction == "left"):
            logger.info(f"{roi_id} → Prioritizing 'up' over 'left' (direction was {best_direction})")
            max_text_pixels = text_pixels
            best_x, best_y, best_w, best_h = new_x, new_y-10, new_w, new_h+15
            best_direction = direction

    # SECOND PASS: Expand RIGHT & LEFT after moving UP
    if best_direction == "up":
        # logger.info("Second pass after UP movement: Expanding RIGHT and LEFT")
        # expand_right_limit = min(width, best_x + best_w + 120)  # Expand right aggressively
        # expand_left_limit = max(0, best_x - 40)  # Expand left slightly
        # best_w = expand_right_limit - expand_left_limit
        # best_x = expand_left_limit
        logger.info("Expanding RIGHT & LEFT after UP to capture full width")
        expand_right_limit = min(width, best_x + best_w + 90)  # Extend more right
        expand_left_limit = max(0, best_x - 70)  # Extend more left
        best_w = expand_right_limit - expand_left_limit
        best_x = expand_left_limit
        expand_top_limit = max(0, best_y - 55)  # Move higher to ensure text is captured
        best_h = (best_y + best_h) - expand_top_limit  # Adjust height
        best_y = expand_top_limit  # Move y upward

    print(f"Final Best Direction: {best_direction}, Text Pixels: {max_text_pixels}")
    return best_x, best_y, best_w, best_h


def extract_text_from_roi(image, roi_x, roi_y, roi_w, roi_h):
    """
    Extracts text from the given ROI in an image using Tesseract OCR.

    :param image: The original image
    :param roi_x: X-coordinate of the ROI
    :param roi_y: Y-coordinate of the ROI
    :param roi_w: Width of the ROI
    :param roi_h: Height of the ROI
    :return: Extracted text
    """
    if roi_w <= 0 or roi_h <= 0:
        logger.error(f"Invalid ROI dimensions: (x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h})")
        return ""

    # Extract ROI from the image
    roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if roi is None or roi.size == 0:
        logger.error(f"Extracted an empty ROI: (x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h})")
        return ""

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Convert to PIL format (fixes "Unsupported image object" error)
    pil_roi = Image.fromarray(gray_roi)

    try:
        # Apply OCR
        extracted_text = pytesseract.image_to_string(pil_roi, config="--psm 6")
    except Exception as e:
        logger.error(f"Error in OCR processing: {e}")
        return ""

    return extracted_text.strip()


# Load the image
image_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/page0.jpg"  # Update this path
image = cv2.imread(image_path)

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

        # Filter by aspect ratio and size
        if 0.8 < aspect_ratio < 1.2 and 10 < w < 50 and 10 < h < 50:  # Likely a checkbox
            # Find the text ROI to the right of the checkbox
            roi_x, roi_y, roi_w, roi_h = find_text_roi(thresh, x, y, w, h)
            extracted_text = extract_text_from_roi(image, roi_x, roi_y, roi_w, roi_h)
            print(f"Extracted Text from ROI ({roi_x}, {roi_y}, {roi_w}, {roi_h}):\n{extracted_text}")
            # Extract the text ROI from the image
            text_roi = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Save the cropped ROI
            cv2.imwrite(f"text_roi_{x}_{y}.jpg", text_roi)

            # Debug: Draw the ROI on the original image
            cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

# Display the annotated image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated image
cv2.imwrite("annotated_image.jpg", image)
