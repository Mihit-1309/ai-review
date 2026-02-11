from sklearn.metrics.pairwise import cosine_similarity
from components.database import topic_store, embedding_cache
from components.embeddings import embed_text


# 🔑 Two thresholds (important)
GENERIC_TO_SPECIFIC_THRESHOLD = 0.50
NORMAL_THRESHOLD = 0.70


# -------------------------
# Helpers
# -------------------------
def normalize_topic(topic: str) -> str:
    return topic.strip().lower()

def aspect_head(topic: str) -> str:
    # take first noun-like token
    return topic.split()[0]

def topic_to_sentence(topic: str) -> str:
    """
    Convert a topic into a sentence for more stable embeddings.
    This is ONLY for embeddings, not for display.
    """
    # head = aspect_head(topic)
    return f"This review discusses the {topic} of the product."


def is_generic(topic: str) -> bool:
    # single-word topics like "battery", "display"
    return len(topic.split()) == 1


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


# -------------------------
# Core logic
# -------------------------
def merge_or_create_topic(WSID: str, product_id: str, topic: str, review_id: str):
    topic = normalize_topic(topic)
    topic_embedding = embed_cached(topic_to_sentence(topic))

    existing_topics = list(topic_store.find({
        "wsid": WSID,
        "product_id": product_id
    }))

    for existing in existing_topics:
        existing_topic = normalize_topic(existing["topic"])

        similarity = cosine_similarity(
            [topic_embedding],
            [existing["embedding"]]
        )[0][0]

        # # 🟢 CASE 1: incoming GENERIC → existing SPECIFIC
        # if is_generic(topic) and not is_generic(existing_topic):
        #     if similarity >= GENERIC_TO_SPECIFIC_THRESHOLD:
        #         topic_store.update_one(
        #             {"_id": existing["_id"]},
        #             {
        #                 "$inc": {"count": 1},
        #                 "$addToSet": {"review_ids": review_id}
        #             }
        #         )
        #         return

        # # 🟢 CASE 2: incoming SPECIFIC → existing GENERIC (🔥 MISSING FIX)
        # if not is_generic(topic) and is_generic(existing_topic):
        #     if similarity >= GENERIC_TO_SPECIFIC_THRESHOLD:
        #         # absorb existing generic INTO incoming specific
        #         topic_store.delete_one({"_id": existing["_id"]})

        #         topic_store.insert_one({
        #             "wsid": WSID,
        #             "product_id": product_id,
        #             "topic": topic,
        #             "embedding": topic_embedding,
        #             "count": existing["count"] + 1,
        #             "review_ids": list(
        #                 set(existing["review_ids"] + [review_id])
        #             )
        #         })
        #         return

        # 🟢 CASE 3: normal semantic merge
        if similarity >= NORMAL_THRESHOLD:
            topic_store.update_one(
                {"_id": existing["_id"]},
                {
                    "$inc": {"count": 1},
                    "$addToSet": {"review_ids": review_id}
                }
            )
            return

    # 🔵 CASE 4: create new topic
    topic_store.insert_one({
        "wsid": WSID,
        "product_id": product_id,
        "topic": topic,
        "embedding": topic_embedding,
        "count": 1,
        "review_ids": [review_id]
    })
