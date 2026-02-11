from sentence_transformers import CrossEncoder
from common.logger import get_logger

logger = get_logger(__name__)

logger.info("Loading reranker model")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
