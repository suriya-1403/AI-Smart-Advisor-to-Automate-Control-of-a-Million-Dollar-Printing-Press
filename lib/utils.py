import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from config import DATASET_FOLDER_PATH, MODEL_NAME, SPLIT_PAGES_PATH, VECTOR_STORE_PATH
from constants import PROMPT_TEMPLATE
from log import get_logger
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from pydantic import ValidationError
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Initialize the logger
logger = get_logger(__name__)

# Initialize the components cache
_components_cache = None

def load_and_split_pdf(file_path=None):
    """
    Load and split a PDF into pages using PyPDFLoader.

    :param file_path: str, optional
        Custom file path to the PDF. Defaults to the dataset PDF path from the config.
    :return: list
        List of pages extracted from the PDF.
    :raises FileNotFoundError:
        If the file does not exist.
    :raises Exception:
        If any other error occurs during the PDF loading and splitting process.
    """

    # Use the provided file_path or default to DATASET_PDF_PATH
    # path_to_pdf = file_path or DATASET_PDF_PATH
    path_to_pdf = file_path
    logger.info(f"Using PDF file path: {path_to_pdf}")

    # Ensure the file exists
    if not os.path.exists(path_to_pdf):
        logger.error(f"PDF file not found at path: {path_to_pdf}")
        raise FileNotFoundError(f"PDF file not found at path: {path_to_pdf}")

    try:
        # Load and split the PDF
        logger.info("Starting to load and split the PDF...")
        loader = PyPDFLoader(file_path=str(path_to_pdf))
        pages = loader.load_and_split()
        logger.info(f"Successfully loaded and split the PDF into {len(pages)} pages.")
        return pages
    except Exception as e:
        logger.exception("An error occurred while loading and splitting the PDF.")
        raise

def load_and_split_pdfs(folder_path=None):
    """
    Load and split all PDF files in a given folder into pages using PyPDFLoader.

    :param folder_path: str, optional
        Custom folder path containing PDFs. Defaults to the dataset folder path from the config.
    :return: list
        List of `Document` objects representing pages from all PDFs.
    """
    path_to_folder = folder_path or DATASET_FOLDER_PATH
    logger.info(f"Using folder path: {path_to_folder}")

    if not os.path.isdir(path_to_folder):
        logger.error(f"Folder not found at path: {path_to_folder}")
        raise FileNotFoundError(f"Folder not found at path: {path_to_folder}")

    pdf_files = list(Path(path_to_folder).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in folder: {path_to_folder}")
        raise FileNotFoundError(f"No PDF files found in folder: {path_to_folder}")

    pdf_documents = []

    try:
        for pdf_file in pdf_files:
            logger.info(f"Processing PDF file: {pdf_file}")
            loader = PyPDFLoader(file_path=str(pdf_file))
            pages = loader.load_and_split()

            for i, page in enumerate(pages):
                # Check if page is already a Document
                if isinstance(page, Document):
                    pdf_documents.append(page)
                else:
                    # Wrap string or other types into Document
                    pdf_documents.append(
                        Document(page_content=str(page), metadata={"file": pdf_file.name, "page": i + 1}))

            logger.info(f"Successfully loaded and split {pdf_file.name} into {len(pages)} pages.")
        return pdf_documents
    except Exception as e:
        logger.exception("An error occurred while loading and splitting the PDFs.")
        raise

def import_or_split_pdf(file_path, save_path=SPLIT_PAGES_PATH):
    """
    Load split PDF pages from a saved file or split the PDF if not available.

    :param file_path: str
        The path to the original PDF.
    :param save_path: str
        The path to the saved split pages.
    :return: list
        The split PDF pages.
    """
    if os.path.exists(save_path):
        logger.info(f"Loading split pages from {save_path}...")
        with open(save_path, "rb") as f:
            pages = pickle.load(f)
        logger.info(f"Loaded {len(pages)} pages from saved file.")
    else:
        logger.info("Split pages not found. Splitting the PDF...")
        pages = load_and_split_pdf(file_path)
        save_pages(pages, save_path)
    return pages

def initialize_components():
    """
    Initialize the LangChain components: model, embeddings, parser, and prompt templates.

    :return: dict
        A dictionary containing the initialized components:
        - **model** (Ollama): The LLM model.
        - **embeddings** (OllamaEmbeddings): The embedding model.
        - **parser** (StrOutputParser): The output parser.
        - **general_prompt** (PromptTemplate): The general-purpose prompt template.
        - **test_case_prompt** (PromptTemplate): The specialized prompt template for generating test cases.
    :raises Exception:
        If any error occurs during initialization.
    """
    try:
        logger.info("Initializing LangChain components...")

        # Initialize the LLM model
        # model = Ollama(model=MODEL_NAME)
        model = OllamaLLM(model=MODEL_NAME)
        logger.info(f"LLM model '{MODEL_NAME}' initialized.")

        # Initialize the embeddings
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        logger.info("Embeddings initialized.")

        # Initialize the output parser
        parser = StrOutputParser()
        logger.info("Output parser initialized.")

        # Initialize the general-purpose prompt template
        general_prompt_template = PROMPT_TEMPLATE
        general_prompt = PromptTemplate(template=general_prompt_template)
        logger.info("General-purpose prompt template initialized.")

        return {
            "model": model,
            "embeddings": embeddings,
            "parser": parser,
            "general_prompt": general_prompt,
        }
    except Exception as error:
        logger.exception("Error during component initialization.")
        raise error

def create_chroma_vector_store(pages, embeddings, persist_directory=VECTOR_STORE_PATH):
    """
    Create a Chroma vector store from the provided pages using embeddings.

    :param pages: list
        List of pages extracted from the PDF.
    :param embeddings: Embeddings
        The embedding model for vectorization.
    :param persist_directory: str
        The directory to store the persistent Chroma database.
    :return: Chroma
        A Chroma vector store object.
    """
    try:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        logger.info("Creating Chroma vector store...")

        # Convert pages into Documents
        documents = [
            Document(page_content=page.page_content, metadata=page.metadata)
            for page in pages
        ]

        # Create the Chroma vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_path),
        )

        # Persist data locally
        # vector_store.persist()
        logger.info(f"Chroma vector store created and persisted at {persist_path}.")
        return vector_store
    except Exception as error:
        logger.exception("An error occurred while creating the Chroma vector store.")
        raise error

def load_or_create_chroma_vector_store(pages, embeddings, persist_directory=VECTOR_STORE_PATH):
    """
    Load an existing Chroma vector store from disk or create a new one if not available.

    :param pages: list
        The list of pages to create the vector store if needed.
    :param embeddings: Embeddings
        The embedding model for vectorization.
    :param persist_directory: str
        The directory to store or load the persistent Chroma database.
    :return: Chroma
        The loaded or newly created Chroma vector store.
    """
    try:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if persist_path.exists():
            logger.info(f"Loading Chroma vector store from {persist_directory}...")
            vector_store = Chroma(persist_directory=str(persist_path), embedding_function=embeddings)
            logger.info("Chroma vector store loaded successfully.")
        else:
            logger.info("Chroma vector store not found. Creating a new one...")
            vector_store = create_chroma_vector_store(pages, embeddings, str(persist_path))
        return vector_store
    except Exception as error:
        logger.exception("An error occurred while loading or creating the Chroma vector store.")
        raise error

def get_cached_components():
    """
    Get cached LangChain components or initialize them if not already cached.

    :return: dict
        A dictionary containing LangChain components.
    """
    global _components_cache
    if _components_cache is None:
        _components_cache = initialize_components()
    return _components_cache

def save_pages(pages, file_path):
    """
    Save the split PDF pages to a file.

    :param pages: list
        The list of split pages.
    :param file_path: str
        The path to save the pages.
    """
    with open(file_path, "wb") as f:
        pickle.dump(pages, f)
    logger.info(f"PDF pages saved to {file_path}.")

def parallel_retrieve_context(vector_store, queries):
    """
    Retrieve contexts for multiple queries in parallel.

    :param vector_store: Chroma
        The vector store to query.
    :param queries: list
        The list of queries to retrieve contexts for.
    :return: list
        The list of retrieved contexts.
    """
    retriever = vector_store.as_retriever()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(retriever.get_relevant_documents, queries))
    return results
