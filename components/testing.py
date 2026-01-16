import os
from pinecone import Pinecone, ServerlessSpec # type: ignore
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX")

# # ❌ DO NOT import this file anywhere else
# if index_name not in [i.name for i in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
#     print("✅ Pinecone index created")
# else:
#     print("ℹ️ Pinecone index already exists")

res = index_name.query(
    vector=[0.0] * 384,
    filter={"WSID": "1gm0fkmh7pht429"},
    top_k=5,
    include_metadata=True
)

for m in res["matches"]:
    print(m["metadata"]["product_id"])
