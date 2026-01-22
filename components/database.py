from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "review_db")
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

reviews_collection = db["reviews"]

# Indexes (VERY IMPORTANT)
reviews_collection.create_index(
    [("wsid", 1), ("product_id", 1)]
)
reviews_collection.create_index("embedded")
  
topic_store = db["topic_store"]
processed_reviews = db["processed_reviews"]
embedding_cache = db["embedding_cache"]

# Indexes (VERY IMPORTANT)
topic_store.create_index(
    [("wsid", 1), ("product_id", 1), ("topic", 1)],
    unique=True
)

processed_reviews.create_index(
    [("review_id", 1)],
    unique=True
)

embedding_cache.create_index(
    [("text", 1)],
    unique=True
)