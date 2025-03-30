# AI Smart Advisor to Automate Control of a Million Dollar Printing Press
An intelligent Retrieval Augmented Generation (RAG) system for querying and analyzing HP test print reports.

# Overview
This branch contains the Hybrid RAG component of the HP Print Report Analysis system. It enables users to query processed print reports using natural language, making it easy to search for specific printing configurations and retrieve relevant documents based on technical parameters.
The complete HP Print Report Analysis system consists of two main components:

Data Preprocessing Pipeline: Transforms unstructured PDF test print reports into structured JSON files (not included in this branch)
Hybrid RAG System: Enables intelligent querying of print report data (this branch)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suriya-1403/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press.git
   cd AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press
   git checkout feature/issue-1-document-scoring
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and running with the required models:
   ```bash
   ollama pull llama3.2
   ollama pull granite3-dense
   ollama pull mistral
   ollama pull deepseek-r1
   ```
## Usage

1. Start the application:
   ```bash
   streamlit run main.py
   ```

2. Upload JSON documents containing printing configuration data.

3. Click "Store JSON in Vector Store" to embed and store documents.

4. Enter queries in the search box:
   - For searches: "Find documents with heavy ink coverage, closest to 220 gsm, and coated media."
   - For questions: "Tell me about the media finish and coating in event 75?"

5. View results or get answers based on the retrieved documents.

## Debug Mode

Enable debug logging by setting the DEBUG environment variable:

```bash
streamlit run main.py -- debug
```

## Vector Database Inspection

View the contents of your ChromaDB collections:

```bash
streamlit run viewChromaDB.py
```
