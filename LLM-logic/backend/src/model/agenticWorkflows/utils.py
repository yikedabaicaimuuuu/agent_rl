from dotenv import load_dotenv
import os
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Validate the required API keys
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in .env file.")


def get_retriever(passage_file="../../../dataset/passages.csv" , persist_directory="faiss_store"):
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings()

    # Check if the FAISS index already exists on disk.
    if os.path.exists(persist_directory):
        print(f"Loading existing FAISS index from '{persist_directory}'...")
        # Allow dangerous deserialization if you trust the source.
        db = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Creating new FAISS index and saving to '{persist_directory}'...")
        # Load your data from CSV.
        ucsc_passage_df = pd.read_csv(passage_file)
        # Create documents from the DataFrame.
        ucsc_passge_data_loader = DataFrameLoader(ucsc_passage_df, page_content_column="passage")
        ucsc_passage_data = ucsc_passge_data_loader.load()
        # Split the data into smaller chunks.
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        docs = text_splitter.split_documents(ucsc_passage_data)
        # Create a FAISS index from the documents.
        db = FAISS.from_documents(docs, embeddings)
        # Save the FAISS index locally for future reuse.
        db.save_local(persist_directory)
    return db.as_retriever()