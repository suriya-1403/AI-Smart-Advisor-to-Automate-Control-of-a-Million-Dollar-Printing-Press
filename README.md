
# AI Smart Advisor to Automate Control of a Million Dollar Printing Press
An intelligent hybrid system for querying and analyzing HP test print reports.

# Overview
This branch contains the Hybrid component of the HP Print Report Analysis system. It enables users to query processed print reports using natural language, making it easy to search for specific printing configurations and retrieve relevant documents based on technical parameters.
The complete HP Print Report Analysis system consists of two main components:

Data Preprocessing Pipeline: Transforms unstructured PDF test print reports into structured JSON files (not included in this branch)
Hybrid RAG System: Enables intelligent querying of print report data (this branch)

## Project Structure

```
.
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Backend container setup
├── Dockerfile.frontend        # React frontend container setup
├── deploy.sh / ldeploy.sh     # Deployment helpers
├── run                        # Optional Unix exec helper
├── requirements.txt           # Python backend dependencies
├── setup.py / pyproject.toml  # Python project metadata
├── LICENSE                    # Project license
├── README.md                  # You're here!
│
├── Frontend/                  # ReactJS-based user interface
│
├── mcp_server/                # FastAPI backend
│   ├── mainWeb.py             # Main FastAPI app entrypoint
│   ├── server.py              # Core server setup
│   ├── settings.py            # Configurations and env setup
│   ├── router.py              # API routing
│   ├── workflows.py           # Request processing pipeline
│   ├── ruleset_tools.py       # Rule-based reasoning
│   ├── document_tools.py      # Document search & filtering
│   ├── eventRetrival_tools.py # Print event-specific logic
│   ├── generalKnowledge_tools.py # General semantic tools
│
├── caddy-logs/                # Reverse proxy logs
├── nginx-frontend.conf        # Frontend reverse proxy config
├── Caddyfile                  # Backend reverse proxy config
└── uv.lock                    # Dependencies lockfile (UV)
```
## Features

- **Document Search** using BM25 + filtering
- **Rule-based Reasoning Engine** for configuration validation
- **Workflow Orchestration** using FastAPI + modular tools
- **Event Retrieval** logic with proximity-aware reranking
- **Frontend** with ReactJS for intuitive user interactions
- **Reverse Proxy Setup** with Nginx or Caddy
- **Dockerized** deployment with isolated builds for frontend/backend

## Installation (Recommended: Docker Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/suriya-1403/AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press.git
cd AI-Smart-Advisor-to-Automate-Control-of-a-Million-Dollar-Printing-Press
```

### 2. Get a Groq Developer API Key

- Sign up at [https://console.groq.com](https://console.groq.com)
- Navigate to API section and generate a key.
- Example key format:
  ```
  gsk_L5G
  ```

### 3. Create a `.env` File

In the root directory, create a `.env` file and add:

```env
groq=gsk_L5G
SSE_HOST=mcp-server
SSE_PORT=8050
```



## Usage

### Run with Docker Compose

```bash
docker compose up --build
```

This will start:
- FastAPI backend
- SSE server
- React frontend
- All services pre-configured and networked


1. Upload JSON documents containing printing configuration data.
2. Query for relevant press events using natural language.
3. Example queries:
   - **Search:** “Find documents with heavy ink coverage, closest to 220 gsm, and coated media.”
   - **Question:** “Tell me about the media finish and coating in event 75?”
4. View structured results and insights in the UI.


## API Endpoints

Sample endpoints exposed via FastAPI:

| Method | Route                 | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/check-json-files`   | Checks document directory status  |
| POST   | `/query`              | Accepts and routes user queries   |
| GET    | `/docs`               | Swagger API docs                  |

## Running Locally Without Docker

If you're a developer running the project manually (without Docker), follow these steps:

---

### 1. Modify Frontend API URLs

In `Frontend/src/components/ChatBot.js`, make these changes:

```js
// Original (for Docker):
const API_URL = `/query`;
const response = await fetch(`/check-json-files`);

// Change to this for local development:
const API_URL = `http://localhost:8000/query`;
const response = await fetch(`http://localhost:8000/check-json-files`);
```

---

### 2. Frontend Environment File

Create a `.env.development` file in the `Frontend/` directory:

```env
REACT_APP_BASE_URL=http://localhost:8000
```

Then start the React app:

```bash
cd Frontend
npm install
npm start
```

---

### 3. Update Backend SSE URL

In `mcp_server/mainWeb.py`, find this section:

```python
SSE_URL = f"http://{os.getenv('SSE_HOST')}:{os.getenv('SSE_PORT')}/sse"
# SSE_URL = "http://localhost:8050/sse"
```

**Change to:**

```python
# SSE_URL = f"http://{os.getenv('SSE_HOST')}:{os.getenv('SSE_PORT')}/sse"
SSE_URL = "http://localhost:8050/sse"
```

---

### Summary

| Component     | Docker Version             | Local Version                      |
|---------------|-----------------------------|------------------------------------|
| Frontend API  | `/query`                    | `http://localhost:8000/query`      |
| SSE URL       | `http://mcp-server:8050/sse`| `http://localhost:8050/sse`        |

These changes ensure compatibility whether you're using Docker or running each service individually.

---
