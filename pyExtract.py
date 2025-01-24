import pdfplumber
import json

def extract_checkboxes_and_text(file_path, output_path):
    """
    Extract checkboxes (rectangles) and their associated text from a PDF.
    Save the results in a structured JSON format.

    Args:
    - file_path (str): Path to the PDF file.
    - output_path (str): Path to save the structured JSON output.
    """
    all_pages_data = []  # List to store data for all pages

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_data = {"page": page_number + 1, "checkbox_text_mapping": []}

            # Extract rectangles (potential checkboxes)
            rectangles = page.objects['rect']

            # Extract words with positions
            words = page.extract_words()

            # Group checkboxes with nearby text
            for rect in rectangles:
                checkbox_center = (rect['x0'] + rect['x1']) / 2  # Center X of the checkbox
                rect_top = rect['top']  # Y-coordinate of the checkbox

                # Find nearby words
                nearby_text = []
                for word in words:
                    word_center = (float(word['x0']) + float(word['x1'])) / 2  # Center X of the word
                    word_top = float(word['top'])  # Y-coordinate of the word

                    # Define proximity thresholds (adjust as needed)
                    if abs(word_center - checkbox_center) < 50 and abs(word_top - rect_top) < 20:
                        nearby_text.append(word['text'])

                # Add the checkbox and associated text to the mapping
                page_data["checkbox_text_mapping"].append({
                    "checkbox": {
                        "x0": rect['x0'], "x1": rect['x1'], "top": rect['top'], "bottom": rect['bottom']
                    },
                    "associated_text": " ".join(nearby_text)
                })

            all_pages_data.append(page_data)

    # Save the structured data to a JSON file
    with open(output_path, "w") as f:
        json.dump(all_pages_data, f, indent=4)

    print(f"Extraction complete! Results saved to {output_path}")

# Paths for input PDF and output JSON
file_path = "/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential/PWP at the Show Prints On Heavy Media Pixelle Speciality Solutions PIXELLE SUPERIOR GLOSS Inkjet Coated Gloss 266 gsm-180lb- 9pt Cover Oct 22.pdf"  # Replace with the path to your PDF file
output_path = "outputExtract.json"  # Replace with the desired output path for JSON

# Run the extraction
extract_checkboxes_and_text(file_path, output_path)
