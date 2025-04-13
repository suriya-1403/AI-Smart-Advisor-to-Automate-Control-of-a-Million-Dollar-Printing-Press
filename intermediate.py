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
def get_media_weight_class(df, weight_gsm):
    df.columns = df.columns.str.strip().str.lower()
    
    for idx, row in df.iterrows():
        try:
            weight_min = float(row.iloc[0])
            weight_max = float(row.iloc[1])
            weight_class = row.iloc[2]

            if weight_min <= weight_gsm <= weight_max:
                return weight_class
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
    "media weight class": "mediaweight_class.html",
}

ruleset_folder = "Rulesets/all_intermediate"
dryer_table_filename = None
pressspeed_table_filename = None
maxpressspeed_table_filename = "Max_Press_Speed.html"

 # Final table for LLM

#  Load & Filter Ruleset Tables 

user_conditions_query = """
media coating - Coated 
media finish - Glossy
ink coverage - 57
optical density - 96
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
                # Dynamically select final table based on ink coverage class
                ink_class_normalized = ink_class.strip().lower()
                final_table_filename = None

                if ink_class_normalized == "light":
                    dryer_table_filename = "Nominal_Dryer_Power_InkCoverageClass_Light.html"
                    pressspeed_table_filename = "Nominal_Press_Speed_InkCoverageClass_Light.html"
                elif ink_class_normalized == "medium":
                    dryer_table_filename = "Nominal_Dryer_Power_InkCoverageClass_Medium.html"
                    pressspeed_table_filename = "Nominal_Press_Speed_InkCoverageClass_Medium.html"
                elif ink_class_normalized == "heavy":
                    dryer_table_filename = "Nominal_Dryer_Power_InkCoverageClass_Heavy.html"
                    pressspeed_table_filename = "Nominal_Press_Speed_InkCoverageClass_Heavy.html"
                else:
                    print("⚠️ Unknown Ink Coverage Class")


            else:
                print(" No match found for Ink Coverage Class.")
        except Exception as e:
            print(f" Error parsing ink coverage values: {e}")
        continue  # Skip normal filter_table for this one

    if key == "media weight class":
        try:
            weight_val = float(conditions.get("media weight", 0))
            weight_class = get_media_weight_class(df, weight_val)
            if weight_class:
                lookup_values[key] = weight_class
                print(f" Media Weight Class matched: {weight_class}")
            else:
                print(" No match found for Media Weight Class.")
        except Exception as e:
            print(f" Error parsing media weight value: {e}")
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

# Load Dryer Power Table
dryer_path = os.path.join(ruleset_folder, dryer_table_filename)
dryer_df = parse_html_to_df(dryer_path)

# Load Press Speed Table
pressspeed_path = os.path.join(ruleset_folder, pressspeed_table_filename)
press_df = parse_html_to_df(pressspeed_path)

if dryer_df.empty or press_df.empty:
    print("❌ One or both final tables failed to load.")
    exit()

# Load Max Press Speed Table
maxpressspeed_path = os.path.join(ruleset_folder, maxpressspeed_table_filename)
maxpress_df = parse_html_to_df(maxpressspeed_path)

if maxpress_df.empty:
    print("❌ Max Press Speed table failed to load.")
    exit()

# Normalize and filter both
dryer_df.columns = dryer_df.columns.str.strip().str.lower()
press_df.columns = press_df.columns.str.strip().str.lower()
maxpress_df.columns = maxpress_df.columns.str.strip().str.lower()


filtered_dryer = dryer_df.copy()
filtered_press = press_df.copy()
max_filtered = maxpress_df.copy()


for col, val in lookup_values.items():
    col_lower = col.lower()
    if col_lower in filtered_dryer.columns:
        filtered_dryer = filtered_dryer[filtered_dryer[col_lower].astype(str).str.lower() == val.lower()]
    if col_lower in filtered_press.columns:
        filtered_press = filtered_press[filtered_press[col_lower].astype(str).str.lower() == val.lower()]


# Extract final values
dryer_power = None
press_speed = None

if not filtered_dryer.empty:
    for col in filtered_dryer.columns:
        if "dryer" in col:
            dryer_power = filtered_dryer.iloc[0][col]
            break

if not filtered_press.empty:
    for col in filtered_press.columns:
        if "speed" in col:
            press_speed = filtered_press.iloc[0][col]
            break
# Extract Max Press Speed
# Extract Max Press Speed from vertical table
max_press_speed = None
try:
    if not maxpress_df.empty:
        mode = lookup_values.get("press quality mode", "").strip().lower()

        for idx, row in maxpress_df.iterrows():
            row_mode = str(row.iloc[0]).strip().lower()
            if row_mode == mode:
                max_press_speed = row.iloc[1]
                break
except Exception as e:
    print(f"⚠️ Failed to extract max press speed: {e}")


# Display results
print("\n⚙️ Final Recommendations:")
print(f"Nominal Dryer Power: {dryer_power}")
print(f"Nominal Press Speed: {press_speed}")
print(f"Max Press Speed: {max_press_speed}")


#   Query the LLM 
def query_llama(dryer_table, press_table, maxpress_table, lookup, question, model_name="llama3"):
    llm = load_llm(model_name)
    template = """
You are a technical assistant for HP PageWide industrial presses. Below are three configuration tables:

1. Nominal Dryer Power Table
2. Nominal Press Speed Table
3. Max Press Speed Table

These map print parameters to recommended power and speed values.

LOOKUP VALUES (extracted from prior rulesets):
{lookup}

NOMINAL DRYER POWER TABLE:
{dryer_table}

NOMINAL PRESS SPEED TABLE:
{press_table}

MAX PRESS SPEED TABLE:
{maxpress_table}

Answer the following question using ONLY the above information:
{question}
"""
    prompt = PromptTemplate(
        input_variables=["lookup", "dryer_table", "press_table", "maxpress_table", "question"],
        template=template
    )

    formatted_prompt = prompt.format(
        lookup=str(lookup),
        dryer_table=str(df_to_json(dryer_table))[:2000],
        press_table=str(df_to_json(press_table))[:2000],
        maxpress_table=str(df_to_json(maxpress_table))[:2000],
        question=question
    )

    return llm.invoke(formatted_prompt)

final_question = """
Given the above lookup values and tables:

Return the recommended values in this format:
- Nominal Dryer Power: <value>
- Nominal Press Speed: <value>
- Max Press Speed: <value>
- Optical Density Class: <value>
- Ink Coverage Class: <value>
- Media Weight Class: <value>
- Media Treatment Class: <value>

"""

# LLM Call
# LLM Call
# Optional: Rename keys for LLM output clarity
formatted_lookup = {
    "Optical Density Class": conditions.get("optical density", "N/A"),
    "Ink Coverage Class": lookup_values.get("ink coverage class", "N/A"),
    "Media Weight Class": lookup_values.get("media weight class", "N/A"),
    "Media Treatment Class": lookup_values.get("media treatment class", "N/A"),
    "Press Quality Mode": lookup_values.get("press quality mode", "N/A"),
}

response = query_llama(filtered_dryer, filtered_press, maxpress_df, formatted_lookup, final_question)
print("\n🧠 LLM Response:\n", response)
