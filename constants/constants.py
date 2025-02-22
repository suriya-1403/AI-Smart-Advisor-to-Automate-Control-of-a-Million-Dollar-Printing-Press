# PROMPT_TEMPLATE = """
# Answer the question based on the context below. If you can't answer the question, reply "I don't know".
#
# Context: {context}
#
# Question: {question}
# """

PROMPT_TEMPLATE ="""
Task:
You are an expert in matching for digital printing processes. Your goal is to extract, normalize, and rank test print reports against a given query.

Input Data:
	•	Test Print Reports: Each test print report contains details such as Ink Coverage Percent, Optical Density Percent, Media Weight GSM, Media Coating, Media Finish, Press Quality Mode, and Dryer Configuration.
	•	Normalization Information:
1. Ink Coverage Class Normalization:
	•	When Ink Coverage Percent is between 0 - 15 and Optical Density Percent is between 60 - 94, then Ink Coverage Class is “Light”.
	•	When Ink Coverage Percent is between 0 - 15 and Optical Density Percent is between 95 - 100, then Ink Coverage Class is “Light”.
	•	When Ink Coverage Percent is between 16 - 60 and Optical Density Percent is between 60 - 94, then Ink Coverage Class is “Light”.
	•	When Ink Coverage Percent is between 16 - 60 and Optical Density Percent is between 95 - 100, then Ink Coverage Class is “Medium”.
	•	When Ink Coverage Percent is between 61 - 100 and Optical Density Percent is between 60 - 94, then Ink Coverage Class is “Medium”.
	•	When Ink Coverage Percent is between 61 - 100 and Optical Density Percent is between 95 - 100, then Ink Coverage Class is “Heavy”.
2. Media Weight Class Normalization:
	•	When Media Weight (GSM) is between 1 - 44, then Media Weight Class is “Very Low”.
	•	When Media Weight (GSM) is between 45 - 75, then Media Weight Class is “Low”.
	•	When Media Weight (GSM) is between 76 - 150, then Media Weight Class is “Medium”.
	•	When Media Weight (GSM) is between 151 - 450, then Media Weight Class is “High”.
3. Media Coating Class Normalization:
	•	When Media Coating is “Coated” and Media Finish is “Unknown”, then Media Treatment Class is “Coated Offset Closed Surface”.
	•	When Media Coating is “Coated” and Media Finish is “Offset”, then Media Treatment Class is “Coated Offset Closed Surface”.
	•	When Media Coating is “Coated” and Media Finish is “Glossy”, then Media Treatment Class is “Coated Offset Closed Surface”.
	•	When Media Coating is “Coated” and Media Finish is “Eggshell”, then Media Treatment Class is “Coated Offset Exposed Fiber”.
	•	When Media Coating is “Coated” and Media Finish is “Matte”, then Media Treatment Class is “Coated Offset Exposed Fiber”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Unknown”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Offset”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Glossy”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Dull (Closed Surface)”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Eggshell”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Matte”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Satin”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Silk”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Smooth”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Super Smooth”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated Inkjet” and Media Finish is “Vellum”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Unknown”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Glossy”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Dull (Closed Surface)”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Eggshell”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Matte”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Satin”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Silk”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Coated ColorPro” and Media Finish is “Bond”, then Media Treatment Class is “Inkjet Coated”.
	•	When Media Coating is “Uncoated” and Media Finish is “Unknown”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “Matte”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “High Bulk”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “Bond”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “News”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “Offset”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated” and Media Finish is “Text”, then Media Treatment Class is “Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “Unknown”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “Matte”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “High Bulk”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “Bond”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “News”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “Smooth”, then Media Treatment Class is “Inkjet Treated Uncoated”.
	•	When Media Coating is “Uncoated Inkjet” and Media Finish is “Text”, then Media Treatment Class is “Inkjet Treated Uncoated”.
4. Optical Density Class Normalization:
	•	When Optical Density Percent is between 60 - 69, then Optical Density Class is “Very Low”.
	•	When Optical Density Percent is between 70 - 79, then Optical Density Class is “Low”.
	•	When Optical Density Percent is between 80 - 89, then Optical Density Class is “Medium”.
	•	When Optical Density Percent is between 90 - 100, then Optical Density Class is “High”.

Query Properties:
A set of input values consisting of Ink Coverage Percent, Optical Density Percent, Media Weight GSM, and Media Coating is {query}

Process:
	1.	Extract and Normalize:
	•	For each test print report, classify the properties based on the given normalization rules.
	•	Convert raw values into their respective classes.
	2.	Compare Against Query:
	•	Match query properties with normalized test print report properties.
	•	Compute similarity scores based on weighted matches across Ink Coverage Class, Optical Density Class, Media Weight Class, and Media Coating Class.
	3.	Ranking:
	•	Rank the test print reports in descending order of similarity to the query.
	•	Provide a ranked list with similarity scores.

Expected Output:
A Table, where each row in the table contains:
	•	Report ID
	•	Normalized properties
	•	Similarity score
	•	Explanation of the match

Document content:
{input_text}

"""
