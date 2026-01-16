from pinecone import Pinecone # type: ignore
import os
from dotenv import load_dotenv
from common.logger import get_logger
from sentence_transformers import SentenceTransformer
load_dotenv()
logger = get_logger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# EMBED_MODEL = "multilingual-e5-large"
model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)

# def embed_text(text: str) -> list:
#     """
#     Convert text to vector embedding using Pinecone Inference
#     """
#     if not text or not text.strip():
#         return []

#     try:
#         response = pc.inference.embed(
#             model=EMBED_MODEL,
#             inputs=[text],
#             input_type="passage"
#         )
#         return response[0]["values"]

#     except Exception as e:
#         logger.error(f"Embedding failed: {e}", exc_info=True)
#         return []

def embed_text(text: str):
    vector = model.encode(text).tolist()
    print("🔍 Embedding dimension:", len(vector))
    return vector
