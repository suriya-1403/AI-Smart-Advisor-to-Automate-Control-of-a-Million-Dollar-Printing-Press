from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the dataset
# DATASET_PDF_PATH = BASE_DIR / "dataset" / "Rulesets" /"PWIGreenButton-Job-Settings-Rules-Mode- 2020-02-27 Reframed for test 2024-10-03 AcrobatSaveAsPDF - HP Confidential.pdf"

DATASET_FOLDER_PATH = BASE_DIR / "dataset" / "sample10"
# Path to the Vector Store
VECTOR_STORE_PATH = str(BASE_DIR / "storage" /"vector_store")

# Path to the Split Page
SPLIT_PAGES_PATH = BASE_DIR / "storage" /"split_pages.pkl"

# Model Name
MODEL_NAME = "llama3.2"
# MODEL_NAME = "deepseek-r1"
# MODEL_NAME = "granite3-dense"
# MODEL_NAME = "qwen2.5-coder"
