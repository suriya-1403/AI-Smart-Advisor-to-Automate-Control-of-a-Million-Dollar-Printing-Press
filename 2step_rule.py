import os
import pandas as pd
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def load_llm(model_name="llama3"):
    return OllamaLLM(model=model_name)

def parse_html_to_df(file_path):
    try:
        tables = pd.read_html(file_path)
        return tables[0] if tables else pd.DataFrame()
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return pd.DataFrame()

def df_to_json(df):
    return df.to_dict(orient="records")

def extract_conditions_from_query(user_query):
    conditions = {}
    lines = user_query.splitlines()
    for line in lines:
        if "-" in line:
            key, val = map(str.strip, line.split("-", 1))
            conditions[key.lower()] = val
    return conditions

def filter_table(df, conditions):
    df.columns = df.columns.str.strip().str.lower()
    for col, val in conditions.items():
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() == val.lower()]
    return df
def get_ink_coverage_class(df, ink_percent, density_percent):
    # Clean column headers
    df.columns = df.columns.str.strip().str.lower()
    
    # Try to infer column positions
    for idx, row in df.iterrows():
        try:
            ink_min = float(row.iloc[0])
            ink_max = float(row.iloc[1])
            density_min = float(row.iloc[2])
            density_max = float(row.iloc[3])
            coverage_class = row.iloc[4]

            if ink_min <= ink_percent <= ink_max and density_min <= density_percent <= density_max:
                return coverage_class
        except Exception:
            continue
    return None

def get_media_treatment_class(df, coating, finish):
    df.columns = df.columns.str.strip().str.title()
    treatment_conditions = {
        "media coating": coating,
        "media finish": finish
    }
    filtered_df = filter_table(df, treatment_conditions)
    print(filtered_df)
    if not filtered_df.empty:
        if "media treatment class" in filtered_df.columns:
            return filtered_df.iloc[0]["media treatment class"]
    return None



#  Define File Mapping

ruleset_files = {
    "media treatment class": "mediatreatment_class.html",
    "ink coverage class": "Ink_coverage-converted.html",
    "media weight class": "mediaweight_class.html"
}

ruleset_folder = "Rulesets/first_order"
final_table_file = "final.html"  # Final table for LLM

#  Load & Filter Ruleset Tables 

user_conditions_query = """
media coating - Coated
media finish - Glossy
ink coverage - 12
optical density - 95
media weight - 50
press quality - Quality
"""

conditions = extract_conditions_from_query(user_conditions_query)

lookup_values = {}

if "press quality" in conditions:
    lookup_values["press quality mode"] = conditions["press quality"]

for key, filename in ruleset_files.items():
    path = os.path.join(ruleset_folder, filename)
    df = parse_html_to_df(path)
    
    df.columns = df.columns.str.strip().str.lower()
    #df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].reset_index(drop=True)
    print(df)
    if df.empty:
        print(f" Could not parse: {filename}")
        continue

    if key == "ink coverage class":
        try:
            ink_percent = float(conditions.get("ink coverage", 0))
            density_percent = float(conditions.get("optical density", 0))
            ink_class = get_ink_coverage_class(df, ink_percent, density_percent)
            if ink_class:
                lookup_values[key] = ink_class
                print(f"Ink Coverage Class matched: {ink_class}")
            else:
                print(" No match found for Ink Coverage Class.")
        except Exception as e:
            print(f" Error parsing ink coverage values: {e}")
        continue  # Skip normal filter_table for this one

    if key == "media treatment class":
        treatment_class = get_media_treatment_class(
            df,
            conditions.get("media coating"),
            conditions.get("media finish")
        )
        if treatment_class:
            lookup_values[key] = treatment_class
            print(f" Media Treatment Class: {treatment_class}")
        else:
            print(" No match found for Media Treatment Class.")
        continue



    filtered_df = filter_table(df, conditions)
    if not filtered_df.empty:
        print(f" Match found in {filename}")
        print(filtered_df)
        value = filtered_df.iloc[0].to_dict()
        # Extract the final class value (assumes it’s the last column)
        lookup_values[key] = list(value.values())[-1]
    else:
        print(f" No match in {filename}")

print("\n🔍 Fetched Lookup Values:")
print(lookup_values)

#  Load Final Table 

final_table_path = os.path.join(ruleset_folder, final_table_file)
final_df = parse_html_to_df(final_table_path)
if final_df.empty:
    print(" Final table not loaded properly.")
    exit()

# Optional: Filter final table using lookup_values
final_df.columns = final_df.columns.str.strip().str.lower()
filtered_final = final_df.copy()
for col, val in lookup_values.items():
    col_lower = col.lower()
    if col_lower in filtered_final.columns:
        filtered_final = filtered_final[filtered_final[col_lower].astype(str).str.lower() == val.lower()]

print("\n📊 Final Filtered Table:")
print(filtered_final)

#   Query the LLM 

def query_llama(final_table, lookup, question, model_name="llama3"):
    llm = load_llm(model_name)
    template = """
You are a technical assistant for HP PageWide industrial presses. Below is a configuration table that maps print parameters to dryer power recommendations.

LOOKUP VALUES (extracted from prior rulesets):
{lookup}

CONFIGURATION TABLE:
{final_table}

Answer the following question using ONLY the above information:
{question}
"""
    prompt = PromptTemplate(
        input_variables=["lookup", "final_table", "question"],
        template=template
    )

    formatted_prompt = prompt.format(
        lookup=str(lookup),
        final_table=str(df_to_json(final_table))[:2000],  # Clip for context
        question=question
    )

    return llm.invoke(formatted_prompt)

# Final user question
final_question = "What is the recommended nominal dryer power percentage range based on these parameters?"

# LLM Call
response = query_llama(filtered_final, lookup_values, final_question)
print("\n🧠 LLM Response:\n", response)
