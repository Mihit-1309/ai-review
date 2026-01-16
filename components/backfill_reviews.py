from components.database import reviews_collection
from components.embedding_worker import embed_single_review
import time

def backfill_reviews():
    print("🚀 Starting MongoDB → Pinecone embedding")

    cursor = reviews_collection.find({"embedded": False})
    count = 0

    for review in cursor:
        embed_single_review(review)
        count += 1

        if count % 50 == 0:
            print(f"✅ Embedded {count} reviews")
            time.sleep(0.5)  # prevent rate limits

    print(f"🎉 Done. Total embedded: {count}")

if __name__ == "__main__":
    backfill_reviews()
