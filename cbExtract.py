# import module
from pdf2image import convert_from_path

# Store Pdf with convert_from_path function
images = convert_from_path('/Users/suriya/Documents/Github/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/Dataset/Sample print reports- HP Confidential/PWP at the Show Prints On Heavy Media Pixelle Speciality Solutions PIXELLE SUPERIOR GLOSS Inkjet Coated Gloss 266 gsm-180lb- 9pt Cover Oct 22.pdf')

for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save('page' + str(i) + '.jpg', 'JPEG')
