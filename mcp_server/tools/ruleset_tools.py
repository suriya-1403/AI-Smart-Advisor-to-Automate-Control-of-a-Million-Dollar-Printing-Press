# """
# Tools for ruleset evaluation.
# """

# import os

# import pandas as pd
# from mcp.server.fastmcp import FastMCP

# from mcp_server.config import RULESET_FILES, RULESETS_DIR

# # Folder containing ruleset HTML files
# ruleset_folder = RULESETS_DIR

# # File mappings
# ruleset_files = RULESET_FILES


# def parse_html_to_df(file_path):
#     """Parse HTML table into pandas DataFrame"""
#     try:
#         tables = pd.read_html(file_path)
#         return tables[0] if tables else pd.DataFrame()
#     except Exception as e:
#         print(f"Error parsing HTML: {e}")
#         return pd.DataFrame()


# def extract_conditions_from_query(user_query):
#     """Extract key-value pairs from query text"""
#     conditions = {}
#     lines = user_query.splitlines()
#     for line in lines:
#         if "-" in line:
#             key, val = map(str.strip, line.split("-", 1))
#             conditions[key.lower()] = val
#     return conditions


# def filter_table(df, conditions):
#     """Filter table using extracted conditions"""
#     df.columns = df.columns.str.strip().str.lower()
#     for col, val in conditions.items():
#         if col in df.columns:
#             df = df[df[col].astype(str).str.lower() == val.lower()]
#     return df


# def get_ink_coverage_class(df, ink_percent, density_percent):
#     """Determine ink coverage class based on percentage values"""
#     df.columns = df.columns.str.strip().str.lower()

#     for idx, row in df.iterrows():
#         try:
#             ink_min = float(row.iloc[0])
#             ink_max = float(row.iloc[1])
#             density_min = float(row.iloc[2])
#             density_max = float(row.iloc[3])
#             coverage_class = row.iloc[4]

#             if (
#                 ink_min <= ink_percent <= ink_max
#                 and density_min <= density_percent <= density_max
#             ):
#                 return coverage_class
#         except Exception:
#             continue
#     return None


# def get_media_weight_class(df, weight_gsm):
#     """Determine media weight class based on GSM value"""
#     df.columns = df.columns.str.strip().str.lower()

#     for idx, row in df.iterrows():
#         try:
#             weight_min = float(row.iloc[0])
#             weight_max = float(row.iloc[1])
#             weight_class = row.iloc[2]

#             if weight_min <= weight_gsm <= weight_max:
#                 return weight_class
#         except Exception:
#             continue
#     return None


# def setup_ruleset_tools(mcp: FastMCP):
#     @mcp.tool()
#     def evaluate_ruleset(query: str) -> str:
#         """
#         Evaluate printing parameters using ruleset tables

#         Args:
#             query: Multi-line string with parameters in format:
#                   media coating - Coated
#                   media finish - Glossy
#                   ink coverage - 57
#                   optical density - 96
#                   media weight - 50
#                   press quality - Quality

#         Returns:
#             String containing the evaluation results
#         """
#         conditions = extract_conditions_from_query(query)

#         if not conditions:
#             return "No valid parameters found in query"

#         lookup_values = {}

#         # Check if ruleset folder exists
#         if not os.path.exists(ruleset_folder):
#             return f"Ruleset folder not found: {ruleset_folder}"

#         # Process press quality
#         if "press quality" in conditions:
#             lookup_values["press quality mode"] = conditions["press quality"]

#         # Process each ruleset file
#         for key, filename in ruleset_files.items():
#             path = os.path.join(ruleset_folder, filename)
#             if not os.path.exists(path):
#                 continue

#             df = parse_html_to_df(path)
#             if df.empty:
#                 continue

#             if key == "ink coverage class":
#                 try:
#                     ink_percent = float(conditions.get("ink coverage", 0))
#                     density_percent = float(conditions.get("optical density", 0))
#                     ink_class = get_ink_coverage_class(df, ink_percent, density_percent)
#                     if ink_class:
#                         lookup_values[key] = ink_class
#                 except Exception:
#                     pass
#                 continue

#             if key == "media weight class":
#                 try:
#                     weight_val = float(conditions.get("media weight", 0))
#                     weight_class = get_media_weight_class(df, weight_val)
#                     if weight_class:
#                         lookup_values[key] = weight_class
#                 except Exception:
#                     pass
#                 continue

#             filtered_df = filter_table(df, conditions)
#             if not filtered_df.empty:
#                 value = filtered_df.iloc[0].to_dict()
#                 # Extract the final class value (assumes it's the last column)
#                 lookup_values[key] = list(value.values())[-1]

#         # Format and return results
#         results = ["Ruleset Evaluation Results:"]
#         for key, value in lookup_values.items():
#             results.append(f"{key.title()}: {value}")

#         if len(results) == 1:
#             return "No matching values found for the provided parameters"

#         return "\n".join(results)

#     @mcp.resource("rulesets://schema")
#     def get_ruleset_schema() -> str:
#         """Get the schema for ruleset parameters"""
#         return """
#         Available parameters:
#         - media_coating: Coated, Uncoated
#         - media_finish: Glossy, Matte
#         - ink_coverage: 0-100
#         - optical_density: 0-100
#         - media_weight: 0-300
#         - press_quality: Draft, Normal, Quality
#         """

import os
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import warnings
from bs4 import MarkupResemblesLocatorWarning
# from input_extracter import extract_inputs_from_natural_query 
from mcp.server.fastmcp import FastMCP
# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate
import numpy as np
import re
from mcp_server.config import LLM_MODEL, GROQ_API

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


# Suppress BS4 warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# --------------------------
# Configuration Agent System
# --------------------------

@dataclass
class AgentResult:
    """Stores the result of an agent's retrieval operation"""
    name: str
    value: Any
    confidence: float = 1.0  # 0.0-1.0 scale
    source: Optional[str] = None
    details: Optional[Dict] = None
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value} (confidence: {self.confidence:.2f})"


class BaseAgent:
    """Base class for all retrieval agents"""
    
    def __init__(self, name: str, ruleset_folder: str):
        self.name = name
        self.ruleset_folder = ruleset_folder
        
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        """Main method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def _load_table(self, filename: str) -> pd.DataFrame:
        """Helper to load an HTML table into a DataFrame"""
        path = os.path.join(self.ruleset_folder, filename)
        try:
            tables = pd.read_html(path)
            df = tables[0] if tables else pd.DataFrame()
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            return df
        except Exception as e:
            print(f"Error loading table {filename}: {e}")
            return pd.DataFrame()


class InkCoverageClassAgent(BaseAgent):
    """Agent for determining ink coverage class"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("ink_coverage_class", ruleset_folder)
        self.table_filename = "Ink_coverage-converted.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(name="ink coverage class", value="medium", confidence=0.1, 
                             source="default", details={"error": "Failed to load table"})
        
        try:
            ink_percent = float(conditions.get("ink coverage", 25))
            density_percent = float(conditions.get("optical density", 90))
            
            for idx, row in df.iterrows():
                try:
                    ink_min = float(row.iloc[0])
                    ink_max = float(row.iloc[1])
                    density_min = float(row.iloc[2])
                    density_max = float(row.iloc[3])
                    coverage_class = row.iloc[4]

                    if ink_min <= ink_percent <= ink_max and density_min <= density_percent <= density_max:
                        return AgentResult(
                            name="ink coverage class", 
                            value=coverage_class, 
                            confidence=0.9,
                            source=self.table_filename,
                            details={"ink_percent": ink_percent, "density_percent": density_percent}
                        )
                except Exception as e:
                    continue
                    
            # No match found
            return AgentResult(
                name="ink coverage class", 
                value="medium", 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "ink_percent": ink_percent, "density_percent": density_percent}
            )
            
        except Exception as e:
            return AgentResult(
                name="ink coverage class", 
                value="medium", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class MediaWeightClassAgent(BaseAgent):
    """Agent for determining media weight class"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("media_weight_class", ruleset_folder)
        self.table_filename = "mediaweight_class.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(name="media weight class", value="medium", confidence=0.1, 
                             source="default", details={"error": "Failed to load table"})
        
        try:
            weight_gsm = float(conditions.get("media weight", 50))
            
            for idx, row in df.iterrows():
                try:
                    weight_min = float(row.iloc[0])
                    weight_max = float(row.iloc[1])
                    weight_class = row.iloc[2]

                    if weight_min <= weight_gsm <= weight_max:
                        return AgentResult(
                            name="media weight class", 
                            value=weight_class, 
                            confidence=0.9,
                            source=self.table_filename,
                            details={"weight_gsm": weight_gsm}
                        )
                except Exception:
                    continue
                    
            # No match found
            return AgentResult(
                name="media weight class", 
                value="medium", 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "weight_gsm": weight_gsm}
            )
            
        except Exception as e:
            return AgentResult(
                name="media weight class", 
                value="medium", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class MediaTreatmentClassAgent(BaseAgent):
    """Agent for determining media treatment class"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("media_treatment_class", ruleset_folder)
        self.table_filename = "mediatreatment_class.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(name="media treatment class", value="standard", confidence=0.1, 
                             source="default", details={"error": "Failed to load table"})
        
        try:
            coating = conditions.get("media coating", "uncoated").lower()
            finish = conditions.get("media finish", "").lower()
            
            # Clean column names to ensure case consistency
            df.columns = df.columns.str.strip().str.lower()
            
            filtered_df = df[
                (df['media coating'].str.lower() == coating) &
                (df['media finish'].str.lower() == finish if finish else True)
            ]
            
            if not filtered_df.empty:
                treatment_class = filtered_df.iloc[0]["media treatment class"]
                return AgentResult(
                    name="media treatment class", 
                    value=treatment_class, 
                    confidence=0.9,
                    source=self.table_filename,
                    details={"coating": coating, "finish": finish}
                )
            
            # Try with just coating if we have no match
            if finish:
                filtered_df = df[df['media coating'].str.lower() == coating]
                if not filtered_df.empty:
                    treatment_class = filtered_df.iloc[0]["media treatment class"]
                    return AgentResult(
                        name="media treatment class", 
                        value=treatment_class, 
                        confidence=0.7,
                        source=self.table_filename,
                        details={"coating": coating, "finish_missing": True}
                    )
            
            # Default fallback
            return AgentResult(
                name="media treatment class", 
                value="standard", 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "coating": coating, "finish": finish}
            )
            
        except Exception as e:
            return AgentResult(
                name="media treatment class", 
                value="standard", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class NominalDryerPowerAgent(BaseAgent):
    """Agent for determining nominal dryer power"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("nominal_dryer_power", ruleset_folder)
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        # Determine table filename based on ink_coverage_class
        ink_class = lookup_values.get("ink coverage class", "medium").lower().strip()
        
        if ink_class == "light":
            table_filename = "Nominal_Dryer_Power_InkCoverageClass_Light.html"
        elif ink_class == "medium":
            table_filename = "Nominal_Dryer_Power_InkCoverageClass_Medium.html"
        elif ink_class == "heavy":
            table_filename = "Nominal_Dryer_Power_InkCoverageClass_Heavy.html"
        else:
            table_filename = "Nominal_Dryer_Power_InkCoverageClass_Medium.html"
            
        df = self._load_table(table_filename)
        if df.empty:
            return AgentResult(name="nominal dryer power", value="50%", confidence=0.1, 
                             source="default", details={"error": f"Failed to load table {table_filename}"})
        
        try:
            # Apply relevant filters based on lookup values
            filtered_df = df.copy()
            
            for col, val in lookup_values.items():
                col_lower = col.lower()
                if col_lower in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col_lower].astype(str).str.lower() == val.lower()]
            
            # Extract dryer power value
            if not filtered_df.empty:
                for col in filtered_df.columns:
                    if "dryer" in col:
                        dryer_power = filtered_df.iloc[0][col]
                        if isinstance(dryer_power, str):
                            dryer_power = dryer_power.replace(" ", "").replace("-%", "%").replace("--", "-").strip()
                        
                        return AgentResult(
                            name="nominal dryer power", 
                            value=dryer_power, 
                            confidence=0.9,
                            source=table_filename,
                            details={"applied_filters": {k: v for k, v in lookup_values.items() if k.lower() in filtered_df.columns}}
                        )
            
            # No match found
            return AgentResult(
                name="nominal dryer power", 
                value="50%", 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "table": table_filename}
            )
            
        except Exception as e:
            return AgentResult(
                name="nominal dryer power", 
                value="50%", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class NominalPressSpeedAgent(BaseAgent):
    """Agent for determining nominal press speed"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("nominal_press_speed", ruleset_folder)
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        # Determine table filename based on ink_coverage_class
        ink_class = lookup_values.get("ink coverage class", "medium").lower().strip()
        
        if ink_class == "light":
            table_filename = "Nominal_Press_Speed_InkCoverageClass_Light.html"
        elif ink_class == "medium":
            table_filename = "Nominal_Press_Speed_InkCoverageClass_Medium.html"
        elif ink_class == "heavy":
            table_filename = "Nominal_Press_Speed_InkCoverageClass_Heavy.html"
        else:
            table_filename = "Nominal_Press_Speed_InkCoverageClass_Medium.html"
            
        df = self._load_table(table_filename)
        if df.empty:
            return AgentResult(name="nominal press speed", value="85%-95%", confidence=0.1, 
                             source="default", details={"error": f"Failed to load table {table_filename}"})
        
        try:
            # Apply relevant filters based on lookup values
            filtered_df = df.copy()
            
            for col, val in lookup_values.items():
                col_lower = col.lower()
                if col_lower in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col_lower].astype(str).str.lower() == val.lower()]
            
            # Extract press speed value
            if not filtered_df.empty:
                for col in filtered_df.columns:
                    if "speed" in col:
                        press_speed = filtered_df.iloc[0][col]
                        
                        return AgentResult(
                            name="nominal press speed", 
                            value=press_speed, 
                            confidence=0.9,
                            source=table_filename,
                            details={"applied_filters": {k: v for k, v in lookup_values.items() if k.lower() in filtered_df.columns}}
                        )
            
            # No match found
            return AgentResult(
                name="nominal press speed", 
                value="85%-95%", 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "table": table_filename}
            )
            
        except Exception as e:
            return AgentResult(
                name="nominal press speed", 
                value="85%-95%", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class MaxPressSpeedAgent(BaseAgent):
    """Agent for determining max press speed"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("max_press_speed", ruleset_folder)
        self.table_filename = "Max_Press_Speed.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(name="max press speed", value=100, confidence=0.1, 
                             source="default", details={"error": "Failed to load table"})
        
        try:
            # Get press quality mode
            mode = conditions.get("press quality", "performance").strip().lower()
            
            for idx, row in df.iterrows():
                row_mode = str(row.iloc[0]).strip().lower()
                if row_mode == mode:
                    max_speed = row.iloc[1]
                    return AgentResult(
                        name="max press speed", 
                        value=max_speed, 
                        confidence=0.9,
                        source=self.table_filename,
                        details={"mode": mode}
                    )
            
            # No match found
            return AgentResult(
                name="max press speed", 
                value=100, 
                confidence=0.3,
                source="default",
                details={"reason": "No matching rule found", "mode": mode}
            )
            
        except Exception as e:
            return AgentResult(
                name="max press speed", 
                value=100, 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class TargetPressSpeedAgent(BaseAgent):
    """Agent for calculating target press speed"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("target_press_speed", ruleset_folder)
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        try:
            # Get required inputs
            nominal_speed_range = lookup_values.get("nominal press speed", "85%-95%")
            max_speed = float(lookup_values.get("max press speed", 100))
            dryer_config = conditions.get("dryer configuration", "default").strip().lower()
            ink_class = lookup_values.get("ink coverage class", "medium").strip().lower()
            
            # Define multipliers
            multipliers = {
                "default": {"light": 0.63, "medium": 0.75, "heavy": 0.80},
                "1-zone": {"light": 0.63, "medium": 0.75, "heavy": 0.80},
                "3-zone": {"light": 1.40, "medium": 1.30, "heavy": 1.15},
            }
            
            # Parse nominal range like "85%-95%"
            if isinstance(nominal_speed_range, str) and "%" in nominal_speed_range:
                parts = nominal_speed_range.strip("%").split("-")
                nominal_min = float(parts[0])
                nominal_max = float(parts[1]) if len(parts) > 1 else nominal_min
            else:
                nominal_min = nominal_max = float(nominal_speed_range)
            
            # DEM and 2-Zone use nominal directly (bounded)
            if dryer_config in ["dem", "2-zone"]:
                final_min = min(nominal_min, max_speed, 100)
                final_max = min(nominal_max, max_speed, 100)
                multiplier = 1.0
            elif dryer_config in multipliers and ink_class in multipliers[dryer_config]:
                multiplier = multipliers[dryer_config][ink_class]
                final_min = min(nominal_min * multiplier, max_speed, 100)
                final_max = min(nominal_max * multiplier, max_speed, 100)
            else:
                # Default fallback
                multiplier = 0.75  # Medium multiplier as default
                final_min = min(nominal_min * multiplier, max_speed, 100)
                final_max = min(nominal_max * multiplier, max_speed, 100)
            
            target_speed = f"{round(final_min)} - {round(final_max)}"
            
            return AgentResult(
                name="target press speed", 
                value=target_speed, 
                confidence=0.9,
                source="computed",
                details={
                    "nominal_min": nominal_min,
                    "nominal_max": nominal_max,
                    "max_speed": max_speed,
                    "dryer_config": dryer_config,
                    "ink_class": ink_class,
                    "multiplier": multiplier
                }
            )
            
        except Exception as e:
            return AgentResult(
                name="target press speed", 
                value="65 - 75", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class TargetDryerPowerAgent(BaseAgent):
    """Agent for determining target dryer power"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("target_dryer_power", ruleset_folder)
        self.table_filename = "Target_Dryer_Power.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        df.columns = ['dryer configuration', 'ink coverage class', 'target dryer power']
        
        if df.empty:
            return AgentResult(name="target dryer power", value="65%", confidence=0.1, 
                             source="default", details={"error": "Failed to load table"})
        
        try:
            dryer_config = conditions.get("dryer configuration", "default").strip().lower()
            ink_class = lookup_values.get("ink coverage class", "medium").strip().lower()
            nominal_power = lookup_values.get("nominal dryer power", "50%")
            
            # Filter the table
            filtered_df = df[
                (df['dryer configuration'].str.lower() == dryer_config) &
                (df['ink coverage class'].str.lower() == ink_class)
            ]
            
            if not filtered_df.empty:
                target_power = filtered_df.iloc[0]['target dryer power']
                return AgentResult(
                    name="target dryer power", 
                    value=target_power, 
                    confidence=0.9,
                    source=self.table_filename,
                    details={
                        "dryer_config": dryer_config,
                        "ink_class": ink_class,
                        "nominal_power": nominal_power
                    }
                )
                
            # No match found
            return AgentResult(
                name="target dryer power", 
                value="65%", 
                confidence=0.3,
                source="default",
                details={
                    "reason": "No matching rule found", 
                    "dryer_config": dryer_config,
                    "ink_class": ink_class
                }
            )
            
        except Exception as e:
            return AgentResult(
                name="target dryer power", 
                value="65%", 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class TensionsAgent(BaseAgent):
    """Agent for determining tension settings"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("tensions", ruleset_folder)
        self.table_filename = "Tensions.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(
                name="tensions", 
                value={
                    "dryer zone tension": "100",
                    "print zone tension": "100",
                    "unwinder zone tension": "100",
                    "rewinder zone tension": "100"
                }, 
                confidence=0.1, 
                source="default", 
                details={"error": "Failed to load table"}
            )
        
        try:
            # Get inputs
            winders_brand = conditions.get("winders brand", "EMT").strip().lower()
            media_weight_class = lookup_values.get("media weight class", "medium").strip().lower()
            
            # Filter the table
            df.columns = df.columns.str.strip().str.lower()
            filtered_df = df[
                (df['media weight class'].str.lower() == media_weight_class) &
                (df['winders brand'].str.lower() == winders_brand)
            ]
            
            if not filtered_df.empty:
                row = filtered_df.iloc[0]
                tensions = {
                    "dryer zone tension": row["dryer zone"],
                    "print zone tension": row["print zone"],
                    "unwinder zone tension": row["unwinder zone"],
                    "rewinder zone tension": row["rewinder zone"]
                }
                
                return AgentResult(
                    name="tensions", 
                    value=tensions, 
                    confidence=0.9,
                    source=self.table_filename,
                    details={
                        "winders_brand": winders_brand,
                        "media_weight_class": media_weight_class
                    }
                )
                
            # No match found
            return AgentResult(
                name="tensions", 
                value={
                    "dryer zone tension": "100",
                    "print zone tension": "100",
                    "unwinder zone tension": "100",
                    "rewinder zone tension": "100"
                }, 
                confidence=0.3,
                source="default",
                details={
                    "reason": "No matching rule found", 
                    "winders_brand": winders_brand,
                    "media_weight_class": media_weight_class
                }
            )
            
        except Exception as e:
            return AgentResult(
                name="tensions", 
                value={
                    "dryer zone tension": "100",
                    "print zone tension": "100",
                    "unwinder zone tension": "100",
                    "rewinder zone tension": "100"
                }, 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class MoisturizerAgent(BaseAgent):
    """Agent for determining moisturizer settings"""
    
    def __init__(self, ruleset_folder: str):
        super().__init__("moisturizer", ruleset_folder)
        self.table_filename = "Moisturizer.html"
    
    def retrieve(self, conditions: Dict[str, Any], lookup_values: Dict[str, Any]) -> AgentResult:
        df = self._load_table(self.table_filename)
        if df.empty:
            return AgentResult(
                name="moisturizer", 
                value={
                    "moisturizer sides": "1",
                    "moisturizer gsm": "1.5"
                }, 
                confidence=0.1, 
                source="default", 
                details={"error": "Failed to load table"}
            )
        
        try:
            # Get inputs
            moisturizer_model = conditions.get("moisturizer model", "weko").strip().lower()
            surfactant = conditions.get("surfactant", "silicon").strip().lower()
            ink_coverage_class = lookup_values.get("ink coverage class", "medium").strip().lower()
            
            # Filter the table
            df.columns = df.columns.str.strip().str.lower()
            filtered_df = df[
                (df['ink coverage class'].str.lower() == ink_coverage_class) &
                (df['moisturizer model'].str.lower() == moisturizer_model) &
                (df['surfactant'].str.lower() == surfactant)
            ]
            
            if not filtered_df.empty:
                row = filtered_df.iloc[0]
                settings = {
                    "moisturizer sides": row["moisturizer sides"],
                    "moisturizer gsm": row["moisturizer gsm"]
                }
                
                return AgentResult(
                    name="moisturizer", 
                    value=settings, 
                    confidence=0.9,
                    source=self.table_filename,
                    details={
                        "moisturizer_model": moisturizer_model,
                        "surfactant": surfactant,
                        "ink_coverage_class": ink_coverage_class
                    }
                )
                
            # No match found
            return AgentResult(
                name="moisturizer", 
                value={
                    "moisturizer sides": "1",
                    "moisturizer gsm": "1.5"
                }, 
                confidence=0.3,
                source="default",
                details={
                    "reason": "No matching rule found", 
                    "moisturizer_model": moisturizer_model,
                    "surfactant": surfactant,
                    "ink_coverage_class": ink_coverage_class
                }
            )
            
        except Exception as e:
            return AgentResult(
                name="moisturizer", 
                value={
                    "moisturizer sides": "1",
                    "moisturizer gsm": "1.5"
                }, 
                confidence=0.1,
                source="default",
                details={"error": str(e)}
            )


class InputExtractionAgent:
    """Agent for extracting structured inputs from user query"""
    
    def __init__(self, default_values: Dict[str, Any] = None):
        self.default_values = default_values or {
            "ink coverage": "25",
            "optical density": "90",
            "media weight": "50",
            "media coating": "uncoated",
            "press quality": "performance",
            "dryer configuration": "default",
            "moisturizer model": "weko",
            "surfactant": "silicon",
            "winders brand": "EMT"
        }
        
    def extract_conditions(self, user_query: str) -> Dict[str, Any]:
        """Extract key-value pairs from user query"""
        conditions = {}
        lines = user_query.splitlines()
        
        for line in lines:
            if "-" in line:
                key, val = map(str.strip, line.split("-", 1))
                conditions[key.lower()] = val
        
        # Fill in defaults for missing values
        for key, default_val in self.default_values.items():
            if key not in conditions:
                conditions[key] = default_val
                
        return conditions


class ConfigurationSystem:
    """Main class that orchestrates the whole process"""
    
    def __init__(self, ruleset_folder: str = "Rulesets/all_intermediate"):
        self.ruleset_folder = ruleset_folder
        self.input_extractor = InputExtractionAgent()
        
        # Initialize all agents
        self.agents = [
            InkCoverageClassAgent(ruleset_folder),
            MediaWeightClassAgent(ruleset_folder),
            MediaTreatmentClassAgent(ruleset_folder),
            NominalDryerPowerAgent(ruleset_folder),
            NominalPressSpeedAgent(ruleset_folder),
            MaxPressSpeedAgent(ruleset_folder),
            TargetPressSpeedAgent(ruleset_folder),
            TargetDryerPowerAgent(ruleset_folder),
            TensionsAgent(ruleset_folder),
            MoisturizerAgent(ruleset_folder)
        ]
        
    def process_query(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        """Process a user query and generate configuration"""
        
        # Step 1: Extract inputs from user query
        conditions = self.input_extractor.extract_conditions(user_query)
        if verbose:
            print("Extracted Conditions:")
            for k, v in conditions.items():
                print(f"  {k}: {v}")
                
        # Step 2: Execute intermediate value retrieval agents
        lookup_values = {}
        agent_results = []
        
        # First pass: Get primary classes (ink coverage, media weight, etc.)
        primary_agents = [agent for agent in self.agents if agent.name in (
            "ink_coverage_class", "media_weight_class", "media_treatment_class"
        )]
        
        for agent in primary_agents:
            result = agent.retrieve(conditions, lookup_values)
            agent_results.append(result)
            
            # Store result for lookup by other agents
            if isinstance(result.value, dict):
                for k, v in result.value.items():
                    lookup_values[k] = v
            else:
                lookup_values[result.name] = result.value
                
            if verbose:
                print(f"✅ {result}")
        
        # Second pass: Get secondary values (nominal dryer power, nominal press speed, max press speed)
        secondary_agents = [agent for agent in self.agents if agent.name in (
            "nominal_dryer_power", "nominal_press_speed", "max_press_speed"
        )]
        
        for agent in secondary_agents:
            result = agent.retrieve(conditions, lookup_values)
            agent_results.append(result)
            
            # Store result for lookup by other agents
            if isinstance(result.value, dict):
                for k, v in result.value.items():
                    lookup_values[k] = v
            else:
                lookup_values[result.name] = result.value
                
            if verbose:
                print(f"✅ {result}")
                
        # Third pass: Get final values (target press speed, target dryer power, tensions, moisturizer)
        final_agents = [agent for agent in self.agents if agent.name in (
            "target_press_speed", "target_dryer_power", "tensions", "moisturizer"
        )]
        
        for agent in final_agents:
            result = agent.retrieve(conditions, lookup_values)
            agent_results.append(result)
            
            # Store result for lookup by other agents
            if isinstance(result.value, dict):
                for k, v in result.value.items():
                    lookup_values[k] = v
            else:
                lookup_values[result.name] = result.value
                
            if verbose:
                print(f"✅ {result}")
        
        # Step 3: Generate final output
        results = {
            "inputs": conditions,
            "intermediate_values": {},
            "final_configuration": {}
        }
        
        # Build intermediate values
        results["intermediate_values"] = {
            "ink_coverage_class": lookup_values.get("ink coverage class"),
            "media_weight_class": lookup_values.get("media weight class"),
            "media_treatment_class": lookup_values.get("media treatment class"),
            "nominal_dryer_power": lookup_values.get("nominal dryer power"),
            "nominal_press_speed": lookup_values.get("nominal press speed"),
            "max_press_speed": lookup_values.get("max press speed")
        }
        
        # Build final configuration
        results["final_configuration"] = {
            "target_press_speed": lookup_values.get("target press speed"),
            "target_dryer_power": lookup_values.get("target dryer power"),
            "dryer_zone_tension": lookup_values.get("dryer zone tension"),
            "print_zone_tension": lookup_values.get("print zone tension"),
            "unwinder_zone_tension": lookup_values.get("unwinder zone tension"),
            "rewinder_zone_tension": lookup_values.get("rewinder zone tension"),
            "moisturizer_sides": lookup_values.get("moisturizer sides"),
            "moisturizer_gsm": lookup_values.get("moisturizer gsm")
        }
        
        # Add agent details for debugging
        if verbose:
            results["agent_details"] = {result.name: result.details for result in agent_results}
            
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a readable report from the results"""
        report = []
        
        # Add header
        report.append("# PageWide Press Configuration Report\n")
        
        # Add inputs section
        report.append("## Input Parameters")
        for key, value in results["inputs"].items():
            report.append(f"* {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Add intermediate values section
        report.append("## Intermediate Values")
        for key, value in results["intermediate_values"].items():
            if value is not None:
                report.append(f"* {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Add final configuration section
        report.append("## Final Configuration")
        for key, value in results["final_configuration"].items():
            if value is not None:
                report.append(f"* {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(report)

#cleaned_intermediate_values = convert_numpy(intermediate_values)

# LLM Integration (Optional)
try:
    

    def load_llm(model_name="llama3.2"):
        """Load a LLM from Ollama"""
        return ChatGroq(model="gemma2-9b-it", api_key=GROQ_API)

    def query_llm(input_params, intermediate_values, final_config, question, model_name="llama3.2"):
        """Query a LLM to get insights about the configuration"""
        llm = load_llm(model_name)
        template = """You are a technical assistant for HP PageWide industrial presses.
    Below is a configuration for a press job:

    INPUT PARAMETERS:
    {input_params}

    INTERMEDIATE VALUES:
    {intermediate_values}

    FINAL CONFIGURATION:
    {final_config}

    Answer the following question using ONLY the above information:
    {question}
    """
        prompt = PromptTemplate(
            input_variables=["input_params", "intermediate_values", "final_config", "question"],
            template=template
        )

        cleaned_input_params = convert_numpy(input_params)
        cleaned_intermediate_values = convert_numpy(intermediate_values)
        cleaned_final_config = convert_numpy(final_config)

        formatted_prompt = prompt.format(
            input_params=json.dumps(cleaned_input_params, indent=2),
            intermediate_values=json.dumps(cleaned_intermediate_values, indent=2),
            final_config=json.dumps(cleaned_final_config, indent=2),
            question=question
        )

        return llm.invoke(formatted_prompt)
except ImportError:
    def load_llm(*args, **kwargs):
        print("LangChain not available. Install with: pip install langchain langchain-ollama")
        return None
        
    def query_llm(*args, **kwargs):
        print("LLM querying not available. Install dependencies first.")
        return "LLM integration not available."

def extract_inputs_from_natural_query(query, model_name="llama3"):
    # Step 1: Build the prompt
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""You are an assistant that extracts structured configuration fields from print job descriptions. 
Given the following user query, extract the following fields into a JSON object with exactly these keys:

- media coating
- media finish
- ink coverage
- optical density
- media weight
- press quality
- dryer configuration
- winders brand
- moisturizer model
- surfactant

If any of the values are not present, use these defaults:
"ink coverage": "25",
"optical density": "90",
"media weight": "50",
"media coating": "uncoated",
"press quality": "performance",
"dryer configuration": "default",
"moisturizer model": "weko",
"surfactant": "silicon",
"winders brand": "EMT",
"media finish": "smooth"

USER QUERY:
{query}

Extracted JSON (only the object, nothing else):
"""
    )

    # Step 2: Call the LLM
    #llm = OllamaLLM(model=model_name)
    llm = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API)
    raw = llm.invoke(prompt.format(query=query))

    # Step 3: Extract JSON object from raw output
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("❌ Could not extract JSON from LLM output.")

    json_str = match.group(0)
    try:
        extracted_json = json.loads(json_str)
    except Exception as e:
        print("❌ Failed to parse JSON. Raw output:")
        print(raw)
        raise e

    # Step 4: Validate and fill missing/null fields with defaults
    default_values = {
        "ink coverage": "25",
        "optical density": "90",
        "media weight": "50",
        "media coating": "uncoated",
        "press quality": "performance",
        "dryer configuration": "default",
        "moisturizer model": "weko",
        "surfactant": "silicon",
        "winders brand": "EMT",
        "media finish": "smooth"
    }

    # Normalize and auto-fill nulls
    complete_json = {}
    for key in default_values:
        val = extracted_json.get(key)
        if val in [None, "null", "None", ""]:
            val = default_values[key]
        complete_json[key] = val

    return complete_json




# Example usage
def setup_ruleset_tools(mcp: FastMCP):
    @mcp.tool()
    def evaluate_ruleset(query: str) -> str:
        # Initialize configuration system
        config_system = ConfigurationSystem(ruleset_folder="AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press/mcp_server/data/rulesets")
        
        # Example query
        # user_query = """
        
        # media coating - Uncoated
        # media finish - Eggshell
        # ink coverage - 96
        # optical density - 88
        # media weight - 372
        # press quality - Performance HDK
        # dryer configuration - 1-Zone
        # winders brand - EMT
        # moisturizer model - Eltex
        # surfactant - Silicon


        # """
        
        # # Process query
        # results = config_system.process_query(user_query, verbose=True)

        # natural_language_query = """
        # I'm printing on uncoated, eggshell-finish media with 96 ink coverage and 88 optical density.
        # The media weighs 372 gsm, and I'm using Performance HDK mode with a 1-Zone dryer.
        # We use EMT winders, Eltex moisturizers, and Silicon as surfactant.
        # """

        # print(query)

        # Extract structured inputs using LLM
        structured_inputs = extract_inputs_from_natural_query(query)

        # Format into expected condition string
        formatted_query = "\n".join([f"{k} - {v}" for k, v in structured_inputs.items()])

        # Now continue as usual
        results = config_system.process_query(formatted_query, verbose=True)
        
        print("hello world!!!!")
        # Print report
        print("\n PrintReports:" + config_system.generate_report(results))
        
        # Optional: Ask LLM for insights
        llm_question = "Explain how the ink coverage and dryer configuration affect the target press speed and what improvements could be made."
        llm_response = query_llm(
            results["inputs"],
            results["intermediate_values"],
            results["final_configuration"],
            llm_question
        )
        print("\nLLM Insights:")
        print(llm_response)
        # return f"\n{results}\n{llm_response}\n"

        return {
            "report": config_system.generate_report(results),
            "llm_insights": str(llm_response)
        }
