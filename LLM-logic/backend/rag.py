from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from IPython.display import Markdown, display

load_dotenv()

app = Flask(__name__)

# Global index variable
pinecone_index = None


def initialize_pinecone():
    global pinecone_index
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("default")
    return pinecone_index


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data['query']
    if pinecone_index is None:
        initialize_pinecone()
    response = pinecone_index.query(queries=[query_text], top_k=5)
    return jsonify(response)





if __name__ == '__main__':
    app.run(debug=True)