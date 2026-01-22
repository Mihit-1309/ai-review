import os
import math
from pinecone import Pinecone # type: ignore
from components.database import reviews_collection
from components.embeddings import embed_text
from dotenv import load_dotenv

load_dotenv()

# ✅ NEW Pinecone client (SDK v2+)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def embed_single_review(review: dict):
    text = f"{safe_str(review.get('review_title'))} {safe_str(review.get('review_text'))}"

    vector = (
        review["review_id"],
        embed_text(text),
        {
            "WSID": safe_str(review.get("wsid")),
            "product_id": safe_str(review.get("product_id")),
            "product_name": safe_str(review.get("product_name")),
            "rating": int(review.get("rating", 0)),
            "review_title": safe_str(review.get("review_title")),
            "review_text": safe_str(review.get("review_text")),
            "text": text
        }
    )

    index.upsert(vectors=[vector])

    reviews_collection.update_one(
        {"review_id": review["review_id"]},
        {"$set": {"embedded": True}}
    )

    print(f"✅ Embedded review {review['review_id']}")
