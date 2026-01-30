from utils import get_retriever
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import pprint
# Data model
class AgentState(TypedDict):
    """Holds the conversation messages and an optional counter for rewrite attempts."""
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # messages: Sequence[BaseMessage] 

    # Optional counter for tracking how many times we've tried rewriting
    remaining_steps: RemainingSteps

class AgenticRag:
    """Flow for an agentic RAG model that decides whether to retrieve documents based on a user question"""
    def __init__(self, retriever):
        # self.searched_docs = []
        self.retriever_tool = create_retriever_tool(
            retriever,
            "UCSC_Information_Retriever",
            "Reterives information related to UCSC i.e University of California, Santa Cruz",
        )
        self.tools = [self.retriever_tool]
        self.workflow = StateGraph(AgentState)
        self.setup_workflow()
    
    def setup_workflow(self):
        # Define the nodes we will cycle between
        self.workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([self.retriever_tool])
        self.workflow.add_node("retrieve", retrieve)  # retrieval
        self.workflow.add_node("rewrite", self.rewrite)  # Re-writing the question
        self.workflow.add_node(
            "generate", self.generate
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not
        self.workflow.add_edge(START, "agent")

        # Decide whether to retrieve
        self.workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        self.workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.grade_documents,
        )
        self.workflow.add_conditional_edges(
            "rewrite",
            self.rewrite_guard,
            {
                "agent": "agent",
                "generate": "generate",
            }
        )
        self.workflow.add_edge("generate", END)
        # Compile
        self.graph = self.workflow.compile()
    
    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")
        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(grade)

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        # Chain
        chain = prompt | llm_with_tool

        messages = state["messages"]

        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
                        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                        Here is the initial question:
                        \n ------- \n
                        {question} 
                        \n ------- \n
                        Formulate an improved question: """,
            )
        ]
        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}


    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content
        
        # self.searched_docs.append(docs)
        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
        # LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
        # Chain
        rag_chain = prompt | llm | StrOutputParser()
        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    def rewrite_guard(state: AgentState) -> str:
        """
        Increments the rewrite_attempts count in the state.
        If rewrite_attempts >= 3, we stop rewriting and route to 'generate'.
        Otherwise, we return 'agent' to try rewriting again.
        """
        # If rewrite_attempts not initialized, set to 0
        return "generate" if state["remaining_steps"] <= 10 else "agent"
    def run(self, query: str, streaming: bool = False):
        from langchain_core.messages import HumanMessage
        # Construct the initial state with the required keys.
        inputs = {
            "messages": [
                ("user", query),
            ]
        }
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")
    
if __name__ == "__main__":
    # Create a retriever
    retriever = get_retriever()
    # Create an AgenticRag instance
    agentic_rag = AgenticRag(retriever)
      
    agentic_rag.run("What are the class sizes for the lower and upper division classes under the engineering school?")
    print("Done!")
  
