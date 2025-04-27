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

def compute_target_press_speed(nominal_speed_range, max_speed, dryer_config, ink_class):
    multipliers = {
        "default": {"light": 0.63, "medium": 0.75, "heavy": 0.80},
        "1-zone": {"light": 0.63, "medium": 0.75, "heavy": 0.80},
        "3-zone": {"light": 1.40, "medium": 1.30, "heavy": 1.15},
    }

    dryer_config = dryer_config.strip().lower()
    ink_class = ink_class.strip().lower()

    try:
        max_speed = float(max_speed)
    except:
        print("⚠️ Could not convert max speed to float.")
        return None

    # Parse nominal range like "85%-95%"
    try:
        if isinstance(nominal_speed_range, str) and "%" in nominal_speed_range:
            parts = nominal_speed_range.strip("%").split("-")
            nominal_min = float(parts[0])
            nominal_max = float(parts[1])
        else:
            nominal_min = nominal_max = float(nominal_speed_range)
    except Exception as e:
        print(f"⚠️ Failed to parse nominal speed range: {nominal_speed_range} -> {e}")
        return None

    # DEM and 2-Zone use nominal directly (bounded)
    if dryer_config in ["dem", "2-zone"]:
        final_min = min(nominal_min, max_speed, 100)
        final_max = min(nominal_max, max_speed, 100)
    elif dryer_config in multipliers and ink_class in multipliers[dryer_config]:
        multiplier = multipliers[dryer_config][ink_class]
        final_min = min(nominal_min * multiplier, max_speed, 100)
        final_max = min(nominal_max * multiplier, max_speed, 100)
    else:
        print("⚠️ No multiplier found for this configuration.")
        return None

    return f"{round(final_min)} - {round(final_max)}"

def get_target_dryer_power(df, dryer_config, ink_class, nominal_dryer_power):
    df.columns = df.columns.str.strip().str.lower()
    try:
        # tensions_path = os.path.join(ruleset_folder, "Tensions.html")
        # tensions_df = parse_html_to_df(tensions_path)

        if tensions_df.empty:
            print(f"⚠️ No tensions table found at {tensions_path}. Skipping tensions lookup.")
        else:
            tensions_df.columns = tensions_df.columns.str.strip().str.lower()

            winders_brand = conditions.get("winders brand", "")
            media_weight_class = lookup_values.get("media weight class", "")

            if winders_brand and media_weight_class:
                tensions_row = tensions_df[
                    (tensions_df['media weight class'].str.lower() == media_weight_class.lower()) &
                    (tensions_df['winders brand'].str.lower() == winders_brand.lower())
                ]
                if not tensions_row.empty:
                    tensions_data = tensions_row.iloc[0]
                    lookup_values["dryer zone tension"] = tensions_data["dryer zone"]
                    lookup_values["print zone tension"] = tensions_data["print zone"]
                    lookup_values["unwinder zone tension"] = tensions_data["unwinder zone"]
                    lookup_values["rewinder zone tension"] = tensions_data["rewinder zone"]
                else:
                    print(f"⚠️ No matching tensions found for {media_weight_class} and {winders_brand}")
            else:
                print(f"⚠️ Missing 'winders brand' or 'media weight class' for tensions lookup.")

    except Exception as e:
        print(f"⚠️ Error loading tensions table: {e}")


def get_winder_zone_settings(df, media_weight_class, winders_brand):
    df.columns = df.columns.str.strip().str.lower()
    media_weight_class = media_weight_class.lower().strip()
    winders_brand = winders_brand.lower().strip()
    
    try:
        filtered_df = df[
            (df['media weight class'].str.lower() == media_weight_class) &
            (df['winders brand'].str.lower() == winders_brand)
        ]
        if not filtered_df.empty:
            row = filtered_df.iloc[0]
            settings = {
                "Dryer Zone": row["dryer zone"],
                "Print Zone": row["print zone"],
                "Unwinder Zone": row["unwinder zone"],
                "Rewinder Zone": row["rewinder zone"]
            }
            return settings
        else:
            print(f"⚠️ No matching entry found for {media_weight_class}, {winders_brand}.")
            return None
    except Exception as e:
        print(f"⚠️ Error finding winder zone settings: {e}")
        return None

def get_moisturizer_settings(df, ink_coverage_class, moisturizer_model, surfactant):
    """
    Lookup Moisturizer Sides and Moisturizer GSM 
    based on Ink Coverage Class, Moisturizer Model, and Surfactant
    """
    try:
        # Normalize
        df.columns = df.columns.str.strip().str.lower()
        ink_coverage_class = ink_coverage_class.lower().strip()
        moisturizer_model = moisturizer_model.lower().strip()
        surfactant = surfactant.lower().strip()

        # Filter
        filtered_df = df[
            (df['ink coverage class'].str.lower() == ink_coverage_class) &
            (df['moisturizer model'].str.lower() == moisturizer_model) &
            (df['surfactant'].str.lower() == surfactant)
        ]

        if not filtered_df.empty:
            row = filtered_df.iloc[0]
            return {
                "Moisturizer Sides": row["moisturizer sides"],
                "Moisturizer GSM": row["moisturizer gsm"]
            }
        else:
            print(f"⚠️ No matching moisturizer settings found for {ink_coverage_class}, {moisturizer_model}, {surfactant}")
            return None

    except Exception as e:
        print(f"⚠️ Error finding moisturizer settings: {e}")
        return None


#  Define File Mapping

ruleset_files = {
    "media treatment class": "mediatreatment_class.html",
    "ink coverage class": "Ink_coverage-converted.html",
    "media weight class": "mediaweight_class.html",
    "target press speed": "Target_Press_Speed_Ruleset.html",
    "target dryer power": "Target_Dryer_Power.html", 
    "tensions": "Tensions.html",
    "moisturizer":"Moisturizer.html",
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
dryer configuration - Default
winders brand - EMT
moisturizer model - weko
surfactant - silicon
"""

conditions = extract_conditions_from_query(user_conditions_query)

lookup_values = {}

if "press quality" in conditions:
    lookup_values["press quality mode"] = conditions["press quality"]

if "dryer configuration" in conditions:
    lookup_values["dryer configuration"] = conditions["dryer configuration"]


for key, filename in ruleset_files.items():
    if key == "target dryer power":
        path = os.path.join(ruleset_folder, filename)
        target_dryer_df = parse_html_to_df(path)
        target_dryer_df.columns = ['dryer configuration', 'ink coverage class', 'target dryer power']
        continue

    if key == "target press speed":
        path = os.path.join(ruleset_folder, filename)
        targetpress_df = parse_html_to_df(path)
        print(f"📥 Loaded Target Press Speed Table: {filename}")
        continue


    if key == "moisturizer":
        # 🆕 Handling the moisturizer table
        path = os.path.join(ruleset_folder, filename)
        moisturizer_df = parse_html_to_df(path)
        moisturizer_df.columns = moisturizer_df.columns.str.strip().str.lower()
        if moisturizer_df.empty:
            print(f"⚠️ Could not parse Moisturizer table: {filename}")
            continue
        
        print(f"📥 Loaded Moisturizer Table: {filename}")
        
        # Fetch inputs
        moisturizer_model = conditions.get("moisturizer model", "")
        surfactant = conditions.get("surfactant", "")
        ink_coverage_class = lookup_values.get("ink coverage class", "")

        if moisturizer_model and surfactant and ink_coverage_class:
            filtered_moisturizer = moisturizer_df[
                (moisturizer_df['ink coverage class'].str.lower() == ink_coverage_class.lower()) &
                (moisturizer_df['moisturizer model'].str.lower() == moisturizer_model.lower()) &
                (moisturizer_df['surfactant'].str.lower() == surfactant.lower())
            ]

            if not filtered_moisturizer.empty:
                moisturizer_data = filtered_moisturizer.iloc[0]
                lookup_values["moisturizer sides"] = moisturizer_data["moisturizer sides"]
                lookup_values["moisturizer gsm"] = moisturizer_data["moisturizer gsm"]
            else:
                print(f"⚠️ No match found in Moisturizer table for {ink_coverage_class}, {moisturizer_model}, {surfactant}")
        else:
            print(f"⚠️ Missing 'moisturizer model', 'surfactant', or 'ink coverage class' to fetch moisturizer settings.")

        continue


    if key == "tensions":
        # 🆕 Handling the tensions table
        path = os.path.join(ruleset_folder, filename)
        tensions_df = parse_html_to_df(path)
        tensions_df.columns = tensions_df.columns.str.strip().str.lower()
        if tensions_df.empty:
            print(f"⚠️ Could not parse Tensions table: {filename}")
            continue
        
        print(f"📥 Loaded Tensions Table: {filename}")
        
        # Fetch winders brand and media weight class
        winders_brand = conditions.get("winders brand", "")
        media_weight_class = lookup_values.get("media weight class", "")

        if media_weight_class and winders_brand:
            tensions_row = tensions_df[
                (tensions_df['media weight class'].str.lower() == media_weight_class.lower()) &
                (tensions_df['winders brand'].str.lower() == winders_brand.lower())
            ]
            if not tensions_row.empty:
                tensions_data = tensions_row.iloc[0]
                lookup_values["dryer zone tension"] = tensions_data["dryer zone"]
                lookup_values["print zone tension"] = tensions_data["print zone"]
                lookup_values["unwinder zone tension"] = tensions_data["unwinder zone"]
                lookup_values["rewinder zone tension"] = tensions_data["rewinder zone"]
            else:
                print(f"⚠️ No match found in Tensions table for {media_weight_class} and {winders_brand}")
        else:
            print(f"⚠️ Missing 'winders brand' or 'media weight class' to fetch tensions settings.")

        continue

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
            if isinstance(dryer_power, str):
                dryer_power = dryer_power.replace(" ", "").replace("-%", "%").replace("--", "-").strip()
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

dryer_config = lookup_values.get("dryer configuration", "default")
ink_class = lookup_values.get("ink coverage class", "medium")

target_press_speed = compute_target_press_speed(
    press_speed,             # e.g. "85%-95%" or just "250"
    max_press_speed,         # e.g. 250
    dryer_config,
    ink_class
)

final_target_dryer_power = get_target_dryer_power(target_dryer_df, dryer_config, ink_class, nominal_dryer_power=dryer_power)


target_table = pd.DataFrame([{
    "Dryer Configuration": dryer_config.title(),
    "Ink Coverage Class": ink_class.title(),
    "Nominal Speed": press_speed,
    "Max Speed": max_press_speed,
    "Target Speed (Computed)": target_press_speed,
    "Target Dryer Power": final_target_dryer_power,
    "Dryer Zone Tension": lookup_values.get("dryer zone tension", "N/A"),
    "Print Zone Tension": lookup_values.get("print zone tension", "N/A"),
    "Unwinder Zone Tension": lookup_values.get("unwinder zone tension", "N/A"),
    "Rewinder Zone Tension": lookup_values.get("rewinder zone tension", "N/A"),
    "Moisturizer Sides": lookup_values.get("moisturizer sides", "N/A"),
    "Moisturizer GSM": lookup_values.get("moisturizer gsm", "N/A")
}])


print("\n⚙️ Final Recommendations:")
print(f"Nominal Dryer Power: {dryer_power}")
print(f"Nominal Press Speed: {press_speed}")
print(f"Max Press Speed: {max_press_speed}")
print(f"Target Press Speed: {target_press_speed}")
print(f"Target Dryer Power: {final_target_dryer_power}")



#   Query the LLM 
def query_llama(dryer_table, press_table, maxpress_table, target_table, target_dryer_table, lookup, question, model_name="llama3"):
    llm = load_llm(model_name)
    template = """You are a technical assistant for HP PageWide industrial presses.
 Below are five configuration tables:

1. Nominal Dryer Power Table
2. Nominal Press Speed Table
3. Max Press Speed Table
4. Target Press Speed Table (computed using multiplier logic)
5. Target Dryer Power Table (lookup)
6. Tension Zone Settings Table (lookup)

These map print parameters to recommended power and speed values.

LOOKUP VALUES (extracted from prior rulesets):
{lookup}

NOMINAL DRYER POWER TABLE:
{dryer_table}

NOMINAL PRESS SPEED TABLE:
{press_table}

MAX PRESS SPEED TABLE:
{maxpress_table}

TARGET PRESS SPEED TABLE:
{target_table}

TARGET DRYER POWER TABLE:
{target_dryer_table}

TENSION ZONE SETTINGS TABLE:
- Dryer Zone Tension
- Print Zone Tension
- Unwinder Zone Tension
- Rewinder Zone Tension
(These tensions are based on Media Weight Class and Winders Brand.)

Answer the following question using ONLY the above information:
{question}
"""
    prompt = PromptTemplate(
        input_variables=["lookup", "dryer_table", "press_table", "maxpress_table", "target_table", "target_dryer_table", "question"],  # ✅ all are STRINGS
        template=template
    )

    formatted_prompt = prompt.format(
        lookup=str(lookup),
        dryer_table=str(df_to_json(dryer_table))[:2000],
        press_table=str(df_to_json(press_table))[:2000],
        maxpress_table=str(df_to_json(maxpress_table))[:2000],
        target_table=str(df_to_json(target_table))[:2000],
        target_dryer_table=str(df_to_json(target_dryer_table))[:2000],
        question=question
    )

    return llm.invoke(formatted_prompt)


final_question = """
Given the above lookup values and tables:

Return the recommended values in this format:
- Nominal Dryer Power: <value>
- Nominal Press Speed: <value>
- Max Press Speed: <value>
- Target Press Speed (Computed): <value>
- Optical Density Class: <value>
- Ink Coverage Class: <value>
- Media Weight Class: <value>
- Media Treatment Class: <value>
- Dryer Zone Tension: <value>
- Print Zone Tension: <value>
- Unwinder Zone Tension: <value>
- Rewinder Zone Tension: <value>
- Moisturizer Sides: <value>
- Moisturizer GSM: <value>

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
    "Target Press Speed (Computed)": target_press_speed,  # ✅ add this
    "Dryer Zone Tension": lookup_values.get("dryer zone tension", "N/A"),
    "Print Zone Tension": lookup_values.get("print zone tension", "N/A"),
    "Unwinder Zone Tension": lookup_values.get("unwinder zone tension", "N/A"),
    "Rewinder Zone Tension": lookup_values.get("rewinder zone tension", "N/A"),
    "Moisturizer Sides": lookup_values.get("moisturizer sides", "N/A"),
    "Moisturizer GSM": lookup_values.get("moisturizer gsm", "N/A")
}

response = query_llama(
    dryer_df,
    press_df,
    maxpress_df,
    target_table,
    target_dryer_df,
    formatted_lookup,
    final_question,
    model_name="llama3"  # ✅ this fixes the crash
)

print("\n🧠 LLM Response:\n", response)
