from components.database import reviews_collection
from components.embedding_worker import embed_single_review

print("🔥 mongo_listener.py started")
print("🔥 Opening change stream...")

pipeline = [{"$match": {"operationType": "insert"}}]

with reviews_collection.watch(pipeline) as stream:
    print("✅ Change stream opened successfully")

    for change in stream:
        print("📥 New insert detected")
        embed_single_review(change["fullDocument"])
