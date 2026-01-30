# LLM Logic + Agentic RAG Integration

A multi-model LLM chat application with an integrated agentic RAG (Retrieval-Augmented Generation) pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend                            │
│              (Next.js - Port 3000)                       │
│   Model Selection: [OpenAI] [Claude] [Gemini] [RAG Agent]│
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Main Backend                           │
│               (Flask - Port 5000)                        │
│  /get_response                                           │
│    ├─ provider=openai  → OpenAI API                     │
│    ├─ provider=claude  → Anthropic API                  │
│    ├─ provider=gemini  → Google API                     │
│    └─ method=rag-agent → RAG Service                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Service                           │
│               (FastAPI - Port 8001)                      │
│  Agentic RAG Pipeline:                                   │
│  - ReasoningAgent (Query Optimization)                   │
│  - RetrievalAgent (FAISS Vector Search)                 │
│  - GenerationAgent (Answer Generation)                   │
│  - EvaluationAgent (Quality Assessment)                  │
│  - LangGraph Router (Multi-step Reasoning)              │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key (required)
- Anthropic API Key (optional, for Claude)
- Google API Key (optional, for Gemini)

### 1. Clone and Setup

```bash
cd agent_rl

# Copy environment file and add your API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **RAG Service**: http://localhost:8001

## Local Development

### Backend (Flask)

```bash
cd LLM-logic/backend
pip install -r requirements.txt
python app.py
```

### Frontend (Next.js)

```bash
cd LLM-logic/frontend
pnpm install
pnpm dev
```

### RAG Service (FastAPI)

```bash
cd rag_service
pip install -r requirements.txt
python main.py
```

## Available Models

### LLM Providers
- **OpenAI**: GPT-4o-mini, GPT-4o, GPT-3.5-turbo
- **Claude**: Claude 3 Opus, Sonnet, Haiku
- **Gemini**: Gemini 1.5 Pro, Flash

### Methods
- **RAG Agent**: Multi-step agentic RAG pipeline with reasoning and evaluation
- **Pro-SLM**: Prolog-based symbolic reasoning
- **RAG**: Simple retrieval-augmented generation
- **Chain of Thought**: Step-by-step reasoning
- **Standard**: Direct LLM query

## Project Structure

```
agent_rl/
├── agent_integration/       # Core RAG agents and pipeline
│   ├── agents/              # Agent implementations
│   │   ├── reasoning_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── generation_agent.py
│   │   ├── evaluation_agent.py
│   │   ├── langgraph_rag.py
│   │   └── RLRouterAgent.py
│   └── vectorstore-hotpot/  # FAISS vector index
│
├── rag_service/             # FastAPI RAG Service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── LLM-logic/
│   ├── backend/             # Flask Backend
│   │   ├── src/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── frontend/            # Next.js Frontend
│       ├── src/
│       └── Dockerfile
│
├── docker-compose.yml       # Deployment configuration
└── .env.example             # Environment template
```

## API Endpoints

### Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_response` | POST | Send message and get LLM response |
| `/new_conversation` | POST | Create new conversation |
| `/conversation/<id>` | GET | Get conversation by ID |
| `/user` | POST | Create new user |
| `/login` | POST | User login |

### RAG Service API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Query the RAG pipeline |

## Customizing the Vector Database

The RAG Service uses FAISS for vector search. To use a different dataset:

1. Build your FAISS index using OpenAI embeddings
2. Update `VECTORSTORE_PATH` in your environment
3. Restart the RAG Service

## License

MIT License
