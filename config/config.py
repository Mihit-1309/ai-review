import os

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

PINECONE_MODEL_NAME = "pinecone/llama-text-embed-v2"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "reviews"

DATA_PATH = "data/"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 0
TOP_K = 7
