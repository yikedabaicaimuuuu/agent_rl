from utils import get_retriever
from typing import List, Literal
from typing_extensions import TypedDict
from langgraph.managed.is_last_step import RemainingSteps
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain import hub
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import pprint
# Router Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["retriever", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a retriever.",
    )
    
# Grader Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Hallucinator Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Answer Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    
# Data model
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    remaining_steps: RemainingSteps

class AdaptiveRag:
    """Flow for an adaptive RAG model that decides whether to retrieve documents based on a user question"""
    def __init__(self):
        self.retriever = get_retriever()
        self.web_search_tool = TavilySearchResults(k=3)
        self.setup_llms()
        self.workflow = StateGraph(GraphState)
        self.setup_workflow()
    
    def setup_llms(self):
        """Initialize LLMs and prompts."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.structured_llm_hallucinator = self.llm.with_structured_output(GradeHallucinations)
        self.structured_llm_answer = self.llm.with_structured_output(GradeAnswer)
        
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at routing a user question to a retriever or web search."),
            ("human", "{question}"),
        ])
        self.question_router = self.route_prompt | self.structured_llm_router

        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of a retrieved document to a user question."),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
        
        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
    
    def setup_workflow(self):
        """Define the workflow structure."""
        self.workflow.add_node("web_search", self.web_search)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)
        
        self.workflow.add_conditional_edges(START, self.route_question, {
            "web_search": "web_search",
            "retriever": "retrieve",
        })
        
        self.workflow.add_conditional_edges("web_search", self.search_guard, {
            "generate": "generate",
            END: END,
        })
        
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {
            "transform_query": "transform_query",
            "generate": "generate", 
        })
        
        self.workflow.add_conditional_edges("transform_query", self.rewrite_guard, {
            "web_search": "web_search",
            "retriever": "retrieve"
        })
        
        self.workflow.add_edge("generate", END)
        self.graph = self.workflow.compile()
    
    def retrieve(self, state):
        """Retrieve documents from retriever."""
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state):
        """Generate answer using retrieved documents."""
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """Grade retrieved documents for relevance."""
        question = state["question"]
        documents = state["documents"]
        filtered_docs = [d for d in documents if self.retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score == "yes"]
        return {"documents": filtered_docs, "question": question}
    
    def transform_query(self, state):
        """Transform the query to a better version."""
        question = state["question"]
        better_question = self.llm.invoke({"question": question})
        return {"question": better_question}
    
    def web_search(self, state):
        """Perform a web search."""
        question = state["question"] + " UCSC"
        docs = self.web_search_tool.invoke({"query": question})
        web_results = Document(page_content="\n".join([d["content"] for d in docs]))
        return {"documents": web_results, "question": question}
    
    def route_question(self, state):
        """Route the question to web search or retriever."""
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        return "web_search" if source.datasource == "web_search" else "retriever"
    
    def decide_to_generate(self, state):
        """Decide whether to generate or transform query based on document relevance."""
        return "generate" if state["documents"] else "transform_query"
    
    def search_guard(self, state):
        return END if state["remaining_steps"] <= 5 else "generate"
    
    def rewrite_guard(self, state):
        return "web_search" if state["remaining_steps"] <= 10 else "retriever"
    
    
    def run(self, query: str):
        """Run the adaptive RAG pipeline with a user query."""
        inputs = {"question": query, "documents": [], "generation": "", "remaining_steps": 10}
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint(value, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")

if __name__ == "__main__":
    adaptive_rag = AdaptiveRag()
    adaptive_rag.run("What are the admission requirements for UCSC?")
