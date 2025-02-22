import os
from pdf2image import convert_from_path

# Define the folder where your PDFs are located
input_folder = '/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential'  # Replace with the path to your folder
output_folder = '/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/SamplePrintReportImages'  # Replace with the path to save images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all PDF files in the input folder
pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

# Convert each PDF to images
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_folder, pdf_file)

    # Convert PDF to images (one image per page)
    images = convert_from_path(pdf_path)

    # Save each page as an image
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'{os.path.splitext(pdf_file)[0]}_page_{i + 1}.png')
        image.save(image_path, 'PNG')  # You can change the format (e.g., PNG, JPEG)
        print(f'Saved: {image_path}')
