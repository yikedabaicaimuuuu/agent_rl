- **[Agentic_RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/):**

  - Use this file if you want to implement the Agentic RAG approach.
  - It contains the flow and methods required to run the **Agentic_RAG** pipeline.

- **[Adaptive_RAG](https://arxiv.org/abs/2403.14403):**

  - Use this file if you want to implement the Adaptive RAG approach.
  - It contains the flow and methods required to run the **Adaptive_RAG** pipeline.

- **[Rat_RAG](https://arxiv.org/abs/2403.05313):**

  - Use this file if you want to implement the **Rat RAG** approach (CoT + RAG).
  - This method applies **Chain of Thought (CoT)** reasoning on the question first and then uses **RAG** to retrieve relevant documents.
  - It contains the flow and methods required to run the **Rat_RAG** pipeline.

- **`utils.py`:**
  - Contains common methods shared by all RAG approaches.
  - **`get_retriever`**:
    - Retrieves context from `passage.csv` and processes it.
    - Saves processed information in the `faiss_store` directory.
    - The appropriate RAG models retrieve context locally if available, otherwise, they fetch it from the relevant passages.
