from app.pi_experiments.run_attacks import run_experiment
from app.RAGAgent import RAGAgent
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import json
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

pi_base_path= Path("app/pi_experiments")

def connect_qdrant_vector_store(embedding_model: str) -> QdrantVectorStore:
    """"
    Create or connect to a Qdrant vector store.

    Args:
        chunks: List of document chunks to store
        embedding_model: Name of the embedding model to use
    Returns:
        QdrantVectorStore instance
    """

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    print(f"Connected to Qdrant collection '{COLLECTION_NAME}'")
    return vector_store

def main():
    # Initialize vector store and model (placeholders)
    vector_store = connect_qdrant_vector_store(EMBEDDING_MODEL)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        reasoning_effort="medium",
        api_key=GROQ_API_KEY, 
    )

    # Create RAG Agent
    agent = RAGAgent(
        vector_store=vector_store,
        model=llm
    )

    # Run attack experiments
    results = run_experiment(agent, n=3)
    print(f"Completed {len(results)} attack experiments.")

    # Process results (e.g., save to file, analyze success rates)
    with open(pi_base_path / "attack_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()