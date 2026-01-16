from components.database import reviews_collection
from components.embeddings import embed_text
from components.vector_store import get_index
import math

def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)

def embed_new_reviews():
    index = get_index()
    reviews = reviews_collection.find({"embedded": False})

    vectors = []

    for r in reviews:
        text = f"{safe_str(r.get('review_title'))} {safe_str(r.get('review_text'))}"
        embedding = embed_text(text)

        vectors.append((
            r["review_id"],
            embedding,
            {
                "WSID": safe_str(r.get("wsid")),
                "product_id": safe_str(r.get("product_id")),
                "product_name": safe_str(r.get("product_name")),  # ✅ FIX
                "rating": int(r.get("rating", 0))
            }
        ))

    if vectors:
        index.upsert(vectors)
        reviews_collection.update_many(
            {"embedded": False},
            {"$set": {"embedded": True}}
        )
        print(f"Embedded {len(vectors)} reviews")
    else:
        print("No new reviews to embed")

if __name__ == "__main__":
    embed_new_reviews()
