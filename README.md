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
   ollama pull mistral
   ```
## Usage

1. Start the application:
   ```bash
   uv run -m mcp_server.server.server
   uv run -m mcp_server.mainWeb
   cd Frontend
   npm start
   ```

2. Upload JSON documents containing printing configuration data.

3. Enter queries in the search box:
   - For searches: "Find documents with heavy ink coverage, closest to 220 gsm, and coated media."
   - For questions: "Tell me about the media finish and coating in event 75?"

4. View results or get answers based on the retrieved documents.

