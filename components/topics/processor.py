from components.vector_store import load_vector_store
from components.database import processed_reviews
from components.topics.extractor import extract_topics
from components.topics.merger import merge_or_create_topic
from common.logger import get_logger

logger = get_logger(__name__)

MAX_PINECONE_K = 10000   # Pinecone hard limit
QUERY_TEXT = "product reviews"


def process_new_reviews(
    WSID: str,
    product_id: str,
    total_limit: int = 15000
):
    vectorstore = load_vector_store()

    remaining = total_limit
    seen_review_ids = set()

    while remaining > 0:
        batch_k = min(MAX_PINECONE_K, remaining)

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": batch_k,
                "filter": {
                    "WSID": str(WSID),           # case-sensitive
                    "product_id": str(product_id)
                }
            }
        )

        docs = retriever.invoke(QUERY_TEXT)

        # 🔍 DEBUG 1 — how many docs Pinecone returned
        # print("DEBUG docs fetched:", len(docs))
        logger.info(
                        f"Retrieved {len(docs)} docs for WSID={WSID}, product_id={product_id}"
                    )
        if not docs:
            break

        new_docs = 0

        for doc in docs:
            review_text = doc.page_content.strip()

            review_id = (
            doc.metadata.get("review_id")   # if you stored it explicitly
            or doc.id                       # Pinecone vector ID (BEST)
            )

            # 🔍 DEBUG 2 — review id
            print("DEBUG review_id:", review_id)

            if review_id in seen_review_ids:
                print("DEBUG skipped (duplicate in same batch)")
                continue

            seen_review_ids.add(review_id)

            if processed_reviews.find_one({"review_id": review_id}):
                print("DEBUG skipped (already processed)")
                continue

            topics = extract_topics(review_text)

            # 🔍 DEBUG 3 — topics extracted by LLM
            print("DEBUG topics extracted:", topics)

            for topic in topics:
                merge_or_create_topic(WSID, product_id, topic,review_id)

            processed_reviews.insert_one({
                "review_id": review_id,
                "wsid": WSID,
                "product_id": product_id
            })

            new_docs += 1

        if new_docs == 0:
            break

        remaining -= new_docs
