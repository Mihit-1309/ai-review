from components.database import reviews_collection

print("🚀 Inserting test review...")

result = reviews_collection.insert_one({
    "review_id": "code_test_001",
    "review_title": "Inserted via Python code",
    "review_text": "Testing MongoDB change stream using Python insert",
    "wsid": "S1",
    "product_id": "P1",
    "product_name": "Test Product",
    "rating": 5,
    "embedded": False
})

print("✅ Inserted document ID:", result.inserted_id)



