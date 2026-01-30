"""
RAG Service - FastAPI wrapper for the RAG pipeline
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add agent_integration to path
AGENT_INTEGRATION_PATH = os.path.join(os.path.dirname(__file__), "..", "agent_integration")
sys.path.insert(0, AGENT_INTEGRATION_PATH)

# Change working directory to agent_integration so relative paths work
os.chdir(AGENT_INTEGRATION_PATH)

# Imports from agent_integration
from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent
from agents.langgraph_rag import run_rag_pipeline

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dspy.evaluate import SemanticF1
import dspy


# Global agents (initialized on startup)
_agents = {}


class QueryRequest(BaseModel):
    question: str
    use_router: bool = True  # Whether to use LangGraph router


class QueryResponse(BaseModel):
    answer: str
    question: str
    success: bool
    error: Optional[str] = None


def init_agents():
    """Initialize all agents and vectorstore"""
    global _agents

    # Environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    VECTORSTORE_PATH = os.getenv(
        "VECTORSTORE_PATH",
        os.path.join(AGENT_INTEGRATION_PATH, "vectorstore-hotpot", "hotpotqa_faiss")
    )

    print(f"Initializing RAG Service...")
    print(f"  - Vectorstore path: {VECTORSTORE_PATH}")

    # Configure DSPy
    dspy.configure(
        lm=dspy.LM(
            model=os.getenv("DSPY_MODEL", "gpt-3.5-turbo"),
            api_base="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(os.getenv("DSPY_MAX_TOKENS", "384")),
            timeout=30,
        )
    )

    # Generation LLM
    gen_llm = ChatOpenAI(
        model=os.getenv("GEN_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        temperature=0.0,
        max_tokens=int(os.getenv("GEN_MAX_TOKENS", "512")),
        timeout=60.0,
    )

    # Evaluation LLM
    eval_llm = ChatOpenAI(
        model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        temperature=0.0,
        max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "1024")),
        timeout=60.0,
    )

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMB_MODEL", "text-embedding-ada-002"),
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
    )

    # Load vectorstore
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"  - Vectorstore loaded successfully")

    # Initialize agents
    semantic_f1_metric = SemanticF1(decompositional=True)

    reasoning_agent = ReasoningAgent()
    evaluation_agent = EvaluationAgent(llm=eval_llm)
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,
        top_k=int(os.getenv("RETR_TOP_K", "5"))
    )
    generation_agent = GenerationAgent(
        llm=gen_llm,
        semantic_f1_metric=semantic_f1_metric
    )

    _agents = {
        "reasoning": reasoning_agent,
        "retrieval": retrieval_agent,
        "generation": generation_agent,
        "evaluation": evaluation_agent,
    }

    print("RAG Service initialized successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    init_agents()
    yield
    # Shutdown
    print("RAG Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="RAG Pipeline API for agentic question answering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents_loaded": len(_agents) > 0
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline with a question
    """
    if not _agents:
        raise HTTPException(status_code=503, detail="Agents not initialized")

    try:
        result = run_rag_pipeline(
            question=request.question,
            retrieval_agent=_agents["retrieval"],
            reasoning_agent=_agents["reasoning"],
            generation_agent=_agents["generation"],
            evaluation_agent=_agents["evaluation"],
            use_router=request.use_router,
            visualize=False
        )

        return QueryResponse(
            answer=result.get("answer", ""),
            question=request.question,
            success=True
        )

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return QueryResponse(
            answer="",
            question=request.question,
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_SERVICE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
