from sklearn.metrics.pairwise import cosine_similarity
from components.database import topic_store, embedding_cache
from components.embeddings import embed_text

SIMILARITY_THRESHOLD = 0.60


def embed_cached(text: str):
    cached = embedding_cache.find_one({"text": text})
    if cached:
        return cached["embedding"]

    embedding = embed_text(text)
    embedding_cache.insert_one({
        "text": text,
        "embedding": embedding
    })
    return embedding


def merge_or_create_topic(WSID: str, product_id: str, topic: str, review_id: str):
    topic_embedding = embed_cached(topic)

    existing_topics = topic_store.find({
        "wsid": WSID,
        "product_id": product_id
    })

    for existing in existing_topics:
        similarity = cosine_similarity(
            [topic_embedding],
            [existing["embedding"]]
        )[0][0]

        if similarity >= SIMILARITY_THRESHOLD:
            topic_store.update_one(
                {"_id": existing["_id"]},
                {
                    "$inc": {"count": 1},
                    "$addToSet": {"review_ids": review_id}
                }
                )
            return

    topic_store.insert_one({
        "wsid": WSID,
        "product_id": product_id,
        "topic": topic,
        "embedding": topic_embedding,
        "count": 1,
        "review_ids": [review_id] 
    })
