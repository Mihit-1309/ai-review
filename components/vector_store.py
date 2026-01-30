import warnings
from typing import List
from pinecone import Pinecone  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore # type: ignore
from dotenv import load_dotenv
from common.custom_exception import CustomException
from common.logger import get_logger
from config.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
# -------------------------------------------------------------------------------------------------

import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "reviews"

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    return pc.Index(INDEX_NAME)


# ✅ ADD THIS FUNCTION
def upsert_vectors(vectors: list):
    """
    vectors = [
        {
            "id": "...",
            "values": [...],
            "metadata": {...}
        }
    ]
    """
    index = get_index()
    index.upsert(vectors)
# ------------------------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# class PineconeInferenceEmbeddings(Embeddings):
#     """
#     Custom Embeddings class using Pinecone's Inference API directly.
#     This fixes the issue where langchain-pinecone sends 'query' instead of 'text'.
#     """
#     def __init__(self, api_key: str, model_name: str = "multilingual-e5-large"):
#         self.pc = Pinecone(api_key=api_key)
#         self.model_name = model_name

#     # def embed_documents(self, texts: List[str]) -> List[List[float]]:
#     #     """Embed search docs (passages)."""
#     #     try:
#     #         # E5 models expect 'input_type': 'passage' for documents
#     #         response = self.pc.inference.embed(
#     #             model=self.model_name,
#     #             inputs=[{"text": text} for text in texts],
#     #             parameters={"input_type": "passage", "truncate": "END"}
#     #         )
#     #         return [record['values'] for record in response.data]
#     #     except Exception as e:
#     #         logger.error(f"Error embedding documents: {e}")
#     #         raise

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """
#         Embed documents in batches.
#         Pinecone model 'multilingual-e5-large' supports max 96 inputs per request.
#         """
#         BATCH_SIZE = 96
#         all_embeddings: List[List[float]] = []

#         try:
#             for i in range(0, len(texts), BATCH_SIZE):
#                 batch_texts = texts[i : i + BATCH_SIZE]

#                 response = self.pc.inference.embed(
#                     model=self.model_name,
#                     inputs=[{"text": text} for text in batch_texts],
#                     parameters={
#                         "input_type": "passage",
#                         "truncate": "END"
#                     }
#                 )

#                 batch_embeddings = [
#                     record["values"] for record in response.data
#                 ]
#                 all_embeddings.extend(batch_embeddings)

#             return all_embeddings

#         except Exception as e:
#             logger.error(f"Error embedding documents: {e}")
#             raise

#     def embed_query(self, text: str) -> List[float]:
#         """Embed query text."""
#         try:
#             # E5 models expect 'input_type': 'query' for queries
#             response = self.pc.inference.embed(
#                 model=self.model_name,
#                 inputs=[{"text": text}],
#                 parameters={"input_type": "query", "truncate": "END"}
#             )
#             return response.data[0]['values']
#         except Exception as e:
#             logger.error(f"Error embedding query: {e}")
#             raise


# def load_vector_store():
#     """
#     Connects to an existing Pinecone index to be used as the vector store.
#     """
#     try:
#         logger.info(f"Connecting to existing Pinecone index: {PINECONE_INDEX_NAME}")
        
#         # Use our custom embedding class
#         embeddings = PineconeInferenceEmbeddings(
#             api_key=PINECONE_API_KEY,
#             model_name="multilingual-e5-large"
#         )
        
#         vectorstore = PineconeVectorStore(
#             index_name=PINECONE_INDEX_NAME, 
#             embedding=embeddings,
#             pinecone_api_key=PINECONE_API_KEY
#         )
        
#         logger.info("Successfully connected to Pinecone vector store.")
#         return vectorstore
            
    # except Exception as e:
    #     error_message = CustomException("Failed to connect to Pinecone vector store", e)
    #     logger.error(str(error_message), exc_info=True)
    #     return None

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore # type: ignore
from pinecone import Pinecone # type: ignore
import os
from common.logger import get_logger

logger = get_logger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "reviews"


def load_vector_store():
    """
    Connects to existing Pinecone index using 384-dim embeddings
    """
    try:
        logger.info(f"Connecting to existing Pinecone index: {INDEX_NAME}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="review_text"
        )

        logger.info("Successfully connected to Pinecone vector store.")
        return vectorstore

    except Exception as e:
        logger.error("Failed to connect to Pinecone vector store", exc_info=True)
        return None


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save to vector store.")
        
        logger.info(f"Uploading documents to Pinecone index: {PINECONE_INDEX_NAME}")
        # Use our custom embedding class
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        db = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY,
            text_key="review_text" 
        )
        
        logger.info("Successfully saved documents to Pinecone.")
        return db

    except Exception as e:
        error_message = CustomException("Failed to save to Pinecone vector store", e)
        logger.error(str(error_message), exc_info=True)
        return None