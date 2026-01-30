from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from utils import get_retriever
import pprint
# Define our state schema for a per-step RAT workflow.
class RATState(TypedDict):
    query: str             # User query
    full_cot: str          # Full chain-of-thought generated initially
    thought_steps: List[str]      # Discrete thought steps (as produced by the LLM)
    revised_steps: List[str]      # Revised versions of each thought step (after retrieval and revision)
    final_answer: str      # Aggregated final answer
    current_step: int      # Index of the thought step being processed
    max_iterations: int    # Maximum number of iterations allowed

class RatRag:
    def __init__(self, retriever):
        # self.searched_docs = []
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.retriever = retriever
        self.workflow = StateGraph(RATState)
        self.setup_workflow()
    
    def setup_workflow(self):
        self.workflow.add_node("generate_initial_cot", self.generate_initial_cot)
        self.workflow.add_node("generate_step_query", self.generate_step_query)
        self.workflow.add_node("retrieve_for_step", self.retrieve_for_step)
        self.workflow.add_node("revise_step", self.revise_step)
        self.workflow.add_node("next_step", self.next_step)
        self.workflow.add_node("aggregate_steps", self.aggregate_steps)

        # Define the edges:
        self.workflow.add_edge(START, "generate_initial_cot")
        self.workflow.add_edge("generate_initial_cot", "generate_step_query")
        self.workflow.add_edge("generate_step_query", "retrieve_for_step")
        self.workflow.add_edge("retrieve_for_step", "revise_step")
        self.workflow.add_edge("revise_step", "next_step")
        self.workflow.add_conditional_edges("next_step", self.more_steps, {"process_next": "generate_step_query", END: "aggregate_steps"})

        # Compile the graph.
        self.graph = self.workflow.compile()
    
    # Helper: Split the full chain-of-thought into discrete steps.
    def split_cot(self, full_text: str) -> List[str]:
        steps = [step.strip() for step in full_text.split("\n") if step.strip()]
        return steps

    # Node 1: Generate initial chain-of-thought for the query.
    def generate_initial_cot(self, state: RATState) -> RATState:
        prompt = (
            f"{state['query']}\n\n"
            "You have access to multiple passages about UCSC, including topics such as admissions, class sizes, academic activities, and general course information. "
            "Please think through what pieces of information would be necessary to answer the query effectively. "
            "Provide your chain of thought as a numbered list, with each step on a new line. "
            "For each step, mention the topic or type of information you believe is needed (for example, 'Admissions requirements', 'Class size data', etc.). "
            "Do not include any extra commentary—only output the numbered steps."
        )
        ai_msg = self.llm.invoke([HumanMessage(content=prompt)])
        state["full_cot"] = ai_msg.content
        state["thought_steps"] = self.split_cot(ai_msg.content)
        state["current_step"] = 0
        state["revised_steps"] = []
        return state

    # Node 2: For the current thought step, generate a retrieval query.
    def generate_step_query(self, state: RATState) -> RATState:
        current = state["thought_steps"][state["current_step"]]
        prompt = (
            f"Given the following thought step:\n\n'{current}'\n\n"
            "Please provide a concise, single-line retrieval query that could help find supporting evidence from UCSC-related passages and documents. "
            "Output only the query."
        )
        ai_msg = self.llm.invoke([HumanMessage(content=prompt)])
        state["thought_steps"][state["current_step"]] = current + "\n[QUERY] " + ai_msg.content.strip()
        return state
    
    def retrieve_for_step(self, state: RATState) -> RATState:
        current = state["thought_steps"][state["current_step"]]
        if "[QUERY]" in current:
            query_part = current.split("[QUERY]", 1)[1].strip()
        else:
            query_part = current 
        docs = self.retriever.invoke(query_part)  # Expecting a list of Document objects.
        # Save each document's content in sourceDocs.
        # self.searched_docs.append([doc.page_content for doc in docs])
        retrieved_text = "\n".join([doc.page_content for doc in docs])
        updated_step = current + "\n[RETRIEVED] " + retrieved_text
        state["thought_steps"][state["current_step"]] = updated_step
        return state

    # Node 4: Revise the current thought step using the retrieved evidence.
    def revise_step(self, state: RATState) -> RATState:
        current = state["thought_steps"][state["current_step"]]
        prompt = (
            f"Below is a thought step along with its retrieved evidence:\n\n{current}\n\n"
            "Please revise this thought step to incorporate the evidence and output only the revised thought step."
        )
        ai_msg = self.llm.invoke([HumanMessage(content=prompt)])
        state["revised_steps"].append(ai_msg.content.strip())
        return state

    # Node 5: Move to the next thought step.
    def next_step(self, state: RATState) -> RATState:
        state["current_step"] += 1
        return state

    # Node 6: Aggregate all revised thought steps into a final answer.
    def aggregate_steps(self, state: RATState) -> RATState:
        aggregated = "\n".join(state["revised_steps"])
        prompt = (
            f"User Query: {state['query']}\n\n"
            f"Revised Thought Steps:\n\n{aggregated}\n\n"
            "Based on the above, generate a final answer that addresses the user's query. "
            "If the revised thought steps provide a clear and complete answer, output that final answer along with the relevant details."
            "If they provide only partial information, start your response with 'Sorry, I couldn't find an exact answer, but here is what I found:' followed by a summary of the available information. "
            "If the revised thought steps are not relevant to the query, output: 'Sorry, I don't know the answer. Please check the website at https://admissions.ucsc.edu/contact-us#contact-information for more details.' "
            "Output only the final answer. Also, include links to the relevant UCSC pages if applicable."
        )
        ai_msg = self.llm.invoke([HumanMessage(content=prompt)])
        state["final_answer"] = ai_msg.content.strip()
        return state

    # Conditional function to check if there are more thought steps to process.
    def more_steps(self, state: RATState) -> str:
        total_steps = len(state["thought_steps"])
        effective_max = min(total_steps, state["max_iterations"])
        # print(f"Current step: {state['current_step']}, Total steps: {total_steps}, Effective max: {effective_max}")
        if state["current_step"] < effective_max:
            return "process_next"
        else:
            print("Max iterations reached.")
            return END
    
    # Function to run the full RAT workflow.
    def run(self, query: str):
        initial_state: RATState = {
            "query": query,
            "full_cot": "",
            "thought_steps": [],
            "revised_steps": [],
            "final_answer": "",
            "current_step": 0,
            "max_iterations": 10,
        }
        # final_state = self.graph.invoke(initial_state)
        # print("Final Answer:")
        # print(final_state["final_answer"])
        for output in self.graph.stream(initial_state):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint(value, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")
            
"""
        __start__
            |
            v
 generate_initial_cot
            |
            v
 generate_step_query
     /           \
    v             v
retrieve_for_step   process_next
    |                      |
    v                      |
revise_step                |
    |                      |
    v                      v
 next_step -------------> end
            |
            v
    aggregate_steps

""" 
if __name__ == "__main__":
    retriever = get_retriever()
    rat_rag = RatRag(retriever= retriever)
    rat_rag.run("Who can be a member of the Boating Club?")
       
       