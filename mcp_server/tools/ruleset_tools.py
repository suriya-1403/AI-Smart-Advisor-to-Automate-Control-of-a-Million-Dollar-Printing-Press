"""
Tools for ruleset evaluation.
"""

import os

import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_server.config import RULESET_FILES, RULESETS_DIR

# Folder containing ruleset HTML files
ruleset_folder = RULESETS_DIR

# File mappings
ruleset_files = RULESET_FILES


def parse_html_to_df(file_path):
    """Parse HTML table into pandas DataFrame"""
    try:
        tables = pd.read_html(file_path)
        return tables[0] if tables else pd.DataFrame()
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return pd.DataFrame()


def extract_conditions_from_query(user_query):
    """Extract key-value pairs from query text"""
    conditions = {}
    lines = user_query.splitlines()
    for line in lines:
        if "-" in line:
            key, val = map(str.strip, line.split("-", 1))
            conditions[key.lower()] = val
    return conditions


def filter_table(df, conditions):
    """Filter table using extracted conditions"""
    df.columns = df.columns.str.strip().str.lower()
    for col, val in conditions.items():
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() == val.lower()]
    return df


def get_ink_coverage_class(df, ink_percent, density_percent):
    """Determine ink coverage class based on percentage values"""
    df.columns = df.columns.str.strip().str.lower()

    for idx, row in df.iterrows():
        try:
            ink_min = float(row.iloc[0])
            ink_max = float(row.iloc[1])
            density_min = float(row.iloc[2])
            density_max = float(row.iloc[3])
            coverage_class = row.iloc[4]

            if (
                ink_min <= ink_percent <= ink_max
                and density_min <= density_percent <= density_max
            ):
                return coverage_class
        except Exception:
            continue
    return None


def get_media_weight_class(df, weight_gsm):
    """Determine media weight class based on GSM value"""
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


def setup_ruleset_tools(mcp: FastMCP):
    @mcp.tool()
    def evaluate_ruleset(query: str) -> str:
        """
        Evaluate printing parameters using ruleset tables

        Args:
            query: Multi-line string with parameters in format:
                  media coating - Coated
                  media finish - Glossy
                  ink coverage - 57
                  optical density - 96
                  media weight - 50
                  press quality - Quality

        Returns:
            String containing the evaluation results
        """
        conditions = extract_conditions_from_query(query)

        if not conditions:
            return "No valid parameters found in query"

        lookup_values = {}

        # Check if ruleset folder exists
        if not os.path.exists(ruleset_folder):
            return f"Ruleset folder not found: {ruleset_folder}"

        # Process press quality
        if "press quality" in conditions:
            lookup_values["press quality mode"] = conditions["press quality"]

        # Process each ruleset file
        for key, filename in ruleset_files.items():
            path = os.path.join(ruleset_folder, filename)
            if not os.path.exists(path):
                continue

            df = parse_html_to_df(path)
            if df.empty:
                continue

            if key == "ink coverage class":
                try:
                    ink_percent = float(conditions.get("ink coverage", 0))
                    density_percent = float(conditions.get("optical density", 0))
                    ink_class = get_ink_coverage_class(df, ink_percent, density_percent)
                    if ink_class:
                        lookup_values[key] = ink_class
                except Exception:
                    pass
                continue

            if key == "media weight class":
                try:
                    weight_val = float(conditions.get("media weight", 0))
                    weight_class = get_media_weight_class(df, weight_val)
                    if weight_class:
                        lookup_values[key] = weight_class
                except Exception:
                    pass
                continue

            filtered_df = filter_table(df, conditions)
            if not filtered_df.empty:
                value = filtered_df.iloc[0].to_dict()
                # Extract the final class value (assumes it's the last column)
                lookup_values[key] = list(value.values())[-1]

        # Format and return results
        results = ["Ruleset Evaluation Results:"]
        for key, value in lookup_values.items():
            results.append(f"{key.title()}: {value}")

        if len(results) == 1:
            return "No matching values found for the provided parameters"

        return "\n".join(results)

    @mcp.resource("rulesets://schema")
    def get_ruleset_schema() -> str:
        """Get the schema for ruleset parameters"""
        return """
        Available parameters:
        - media_coating: Coated, Uncoated
        - media_finish: Glossy, Matte
        - ink_coverage: 0-100
        - optical_density: 0-100
        - media_weight: 0-300
        - press_quality: Draft, Normal, Quality
        """
