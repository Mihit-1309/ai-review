import os
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from components.embeddings import embed_text
from components.llm import load_llm
from common.logger import get_logger

logger = get_logger(__name__)
# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

EMBEDDING_DIM = 384  # must match Pinecone index

llm = load_llm()

# --------------------------------------------------
# Prompt
# --------------------------------------------------

CHAT_PROMPT = """
You are an assistant that answers questions by summarizing customer reviews.

STRICT RULES (VERY IMPORTANT):
- Do NOT list keywords
- Do NOT use bullet points
- Do NOT quote reviews
- Do NOT copy phrases verbatim
- Write ONE clear, concise paragraph
- Paraphrase everything in your own words
- Sound natural and professional

If the reviews do not clearly answer the question, say:
"Customers have not clearly mentioned this in their reviews."

Customer reviews:
{context}

User question:
{question}

Write a clear paragraph answering the question:
"""


prompt = PromptTemplate(
    template=CHAT_PROMPT,
    input_variables=["context", "question"]
)

parser = StrOutputParser()


def chat_with_reviews(wsid: str, product_id: str, question: str):

    logger.info(
        "CHAT_REQUEST | wsid=%s | product_id=%s | question=%s",
        wsid, product_id, question
    )

    # --------------------------------------------------
    # Step 1: Create embedding for question
    # --------------------------------------------------
    logger.info("Generating embedding for user question")
    query_embedding = embed_text(question)

    # --------------------------------------------------
    # Step 2: Query Pinecone
    # --------------------------------------------------
    res = index.query(
        vector=query_embedding,
        top_k=5,
        filter={
            "WSID": str(wsid),
            "product_id": str(product_id)
        },
        include_metadata=True
    )

    logger.info(
        "Pinecone query completed | fetched_docs=%d",
        len(res.matches)
    )

    # --------------------------------------------------
    # Step 3: Extract reviews
    # --------------------------------------------------
    reviews_for_llm = []
    reviews_for_ui = []

    for i, m in enumerate(res.matches, start=1):
        review_text = m.metadata.get("review_text") or m.metadata.get("text")
        rating = m.metadata.get("rating")

        logger.info(
            "DOC_%d | score=%.4f | has_text=%s",
            i,
            (m.score or 0.0),
            bool(review_text)
        )

        if review_text:
            reviews_for_llm.append(review_text)
            reviews_for_ui.append({
                "review_text": review_text,
                "rating": rating,
                "product_name": m.metadata.get("product_name")
            })

    logger.info(
        "Review extraction completed | usable_reviews=%d",
        len(reviews_for_llm)
    )

    if not reviews_for_llm:
        logger.info("No usable reviews found for LLM context")
        return {
            "answer": "No reviews found for this product.",
            "reviews": []
        }

    # --------------------------------------------------
    # Step 4: Build context for LLM
    # --------------------------------------------------
    context = "\n\n".join(reviews_for_llm)
    logger.info(
        "Context built for LLM | reviews_used=%d | context_chars=%d",
        len(reviews_for_llm),
        len(context)
    )

    # --------------------------------------------------
    # Step 5: Call LLM
    # --------------------------------------------------
    logger.info("Calling LLM to generate answer")

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": question
    }).strip()

    logger.info("LLM response received")

    # --------------------------------------------------
    # Step 6: Validate answer
    # --------------------------------------------------
    if not answer or len(answer.split()) < 5:
        logger.info(
            "LLM answer weak or empty | returning fallback message"
        )
        answer = "Customers have not clearly mentioned this in their reviews."

    logger.info("CHAT_RESPONSE_READY")

    return {
        "answer": answer,
        "reviews": reviews_for_ui
    }

# --------------------------------------------------------------------------------------------------------------------------------

# ##ONLY RERANKING

# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone # type: ignore
# from components.reranker import reranker

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from components.embeddings import embed_text
# from components.llm import load_llm
# from common.logger import get_logger

# logger = get_logger(__name__)
# # --------------------------------------------------
# # Setup
# # --------------------------------------------------

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(os.getenv("PINECONE_INDEX"))

# EMBEDDING_DIM = 384  # must match Pinecone index

# llm = load_llm()

# # --------------------------------------------------
# # Prompt
# # --------------------------------------------------

# CHAT_PROMPT = """
# You are an assistant that answers questions by summarizing customer reviews.

# STRICT RULES (VERY IMPORTANT):
# - Do NOT list keywords
# - Do NOT use bullet points
# - Do NOT quote reviews
# - Do NOT copy phrases verbatim
# - Write ONE clear, concise paragraph
# - Paraphrase everything in your own words
# - Sound natural and professional

# If the reviews do not clearly answer the question, say:
# "Customers have not clearly mentioned this in their reviews."

# Customer reviews:
# {context}

# User question:
# {question}

# Write a clear paragraph answering the question:
# """


# prompt = PromptTemplate(
#     template=CHAT_PROMPT,
#     input_variables=["context", "question"]
# )

# parser = StrOutputParser()


# def chat_with_reviews(wsid: str, product_id: str, question: str):

#     logger.info(
#         "CHAT_REQUEST | wsid=%s | product_id=%s | question=%s",
#         wsid, product_id, question
#     )

#     # --------------------------------------------------
#     # Step 1: Create embedding for question
#     # --------------------------------------------------
#     logger.info("Generating embedding for user question")
#     query_embedding = embed_text(question)

#     # --------------------------------------------------
#     # Step 2: Query Pinecone
#     # --------------------------------------------------
#     res = index.query(
#         vector=query_embedding,
#         top_k=20,
#         filter={
#             "WSID": str(wsid),
#             "product_id": str(product_id)
#         },
#         include_metadata=True
#     )

#     logger.info(
#         "Pinecone query completed | fetched_docs=%d",
#         len(res.matches)
#     )

#     # --------------------------------------------------
#     # Step 3: Extract reviews
#     # --------------------------------------------------
#     reviews_for_llm = []
#     reviews_for_ui = []

#     for i, m in enumerate(res.matches, start=1):
#         review_text = m.metadata.get("review_text") or m.metadata.get("text")
#         rating = m.metadata.get("rating")

#         logger.info(
#             "DOC_%d | score=%.4f | has_text=%s",
#             i,
#             (m.score or 0.0),
#             bool(review_text)
#         )

#         if review_text:
#             reviews_for_llm.append(review_text)
#             reviews_for_ui.append({
#                 "review_text": review_text,
#                 "rating": rating,
#                 "product_name": m.metadata.get("product_name")
#             })

#     logger.info(
#         "Review extraction completed | usable_reviews=%d",
#         len(reviews_for_llm)
#     )

#     if not reviews_for_llm:
#         logger.info("No usable reviews found for LLM context")
#         return {
#             "answer": "No reviews found for this product.",
#             "reviews": []
#         }
    
#     # --------------------------------------------------
#     # Step 3.5: Rerank reviews using cross-encoder
#     # --------------------------------------------------
#     logger.info(
#         "Starting reranking | candidate_reviews=%d",
#         len(reviews_for_llm)
#     )

#     pairs = [
#         (question, review)
#         for review in reviews_for_llm
#     ]

#     scores = reranker.predict(pairs)

#     scored_reviews = list(zip(reviews_for_llm, scores))

#     # Sort reviews by reranker score (descending)
#     scored_reviews.sort(key=lambda x: x[1], reverse=True)

#     TOP_N = 5
#     # MIN_SCORE = 0.02  # optional but recommended

#     reranked_reviews = [
#         review
#         for review, _ in scored_reviews
#         # if score >= MIN_SCORE
#     [:TOP_N]]

#     for i, (review, score) in enumerate(scored_reviews[:5], start=1):
#         logger.info("RERANK_%d | score=%.4f | text_preview=%s",
#                 i, score, review[:60])


#     # if not reranked_reviews:
#     #     logger.warning(
#     #         "Reranker returned 0 results | falling back to top Pinecone reviews"
#     #     )
#     #     reranked_reviews = reviews_for_llm[:TOP_N]

#     # logger.info(
#     #     "Reranking completed | selected_reviews=%d",
#     #     len(reranked_reviews)
#     # )

#     # --------------------------------------------------
#     # Step 4: Build context for LLM
#     # --------------------------------------------------
#     context = "\n\n".join(reranked_reviews)
#     logger.info(
#         "Context built for LLM | reviews_used=%d | context_chars=%d",
#         len(reviews_for_llm),
#         len(context)
#     )

#     # --------------------------------------------------
#     # Step 5: Call LLM
#     # --------------------------------------------------
#     logger.info("Calling LLM to generate answer")

#     answer = (
#         prompt
#         | llm
#         | StrOutputParser()
#     ).invoke({
#         "context": context,
#         "question": question
#     }).strip()

#     logger.info("LLM response received")

#     # --------------------------------------------------
#     # Step 6: Validate answer
#     # --------------------------------------------------
#     if not answer or len(answer.split()) < 5:
#         logger.info(
#             "LLM answer weak or empty | returning fallback message"
#         )
#         answer = "Customers have not clearly mentioned this in their reviews."

#     logger.info("CHAT_RESPONSE_READY")

#     reranked_set = set(reranked_reviews)

#     filtered_ui_reviews = [
#         r for r in reviews_for_ui
#         if r["review_text"] in reranked_set
#     ]

#     return {
#         "answer": answer,
#         "reviews": filtered_ui_reviews
#     }

# --------------------------------------------------------------------------------------------------------------------------------



# ## RERANKING WITH FILTERING


# import os
# from dotenv import load_dotenv
# from pinecone import Pinecone # type: ignore
# from components.reranker import reranker

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from components.embeddings import embed_text
# from components.llm import load_llm
# from common.logger import get_logger

# logger = get_logger(__name__)
# # --------------------------------------------------
# # Setup
# # --------------------------------------------------

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(os.getenv("PINECONE_INDEX"))

# EMBEDDING_DIM = 384  # must match Pinecone index

# llm = load_llm()

# # --------------------------------------------------
# # Prompt
# # --------------------------------------------------

# CHAT_PROMPT = """
# You are an assistant that answers questions by summarizing customer reviews.

# STRICT RULES (VERY IMPORTANT):
# - Do NOT list keywords
# - Do NOT use bullet points
# - Do NOT quote reviews
# - Do NOT copy phrases verbatim
# - Write ONE clear, concise paragraph
# - Paraphrase everything in your own words
# - Sound natural and professional

# If the reviews do not clearly answer the question, say:
# "Customers have not clearly mentioned this in their reviews."

# Customer reviews:
# {context}

# User question:
# {question}

# Write a clear paragraph answering the question:
# """


# prompt = PromptTemplate(
#     template=CHAT_PROMPT,
#     input_variables=["context", "question"]
# )

# parser = StrOutputParser()


# def chat_with_reviews(wsid: str, product_id: str, question: str):

#     logger.info(
#         "CHAT_REQUEST | wsid=%s | product_id=%s | question=%s",
#         wsid, product_id, question
#     )

#     # --------------------------------------------------
#     # Step 1: Create embedding for question
#     # --------------------------------------------------
#     logger.info("Generating embedding for user question")
#     query_embedding = embed_text(question)

#     # --------------------------------------------------
#     # Step 2: Query Pinecone
#     # --------------------------------------------------
#     res = index.query(
#         vector=query_embedding,
#         top_k=20,
#         filter={
#             "WSID": str(wsid),
#             "product_id": str(product_id)
#         },
#         include_metadata=True
#     )

#     logger.info(
#         "Pinecone query completed | fetched_docs=%d",
#         len(res.matches)
#     )

#     # --------------------------------------------------
#     # Step 3: Extract reviews
#     # --------------------------------------------------
#     reviews_for_llm = []
#     reviews_for_ui = []

#     for i, m in enumerate(res.matches, start=1):
#         review_text = m.metadata.get("review_text") or m.metadata.get("text")
#         rating = m.metadata.get("rating")

#         logger.info(
#             "DOC_%d | score=%.4f | has_text=%s",
#             i,
#             (m.score or 0.0),
#             bool(review_text)
#         )

#         if review_text:
#             reviews_for_llm.append(review_text)
#             reviews_for_ui.append({
#                 "review_text": review_text,
#                 "rating": rating,
#                 "product_name": m.metadata.get("product_name")
#             })

#     logger.info(
#         "Review extraction completed | usable_reviews=%d",
#         len(reviews_for_llm)
#     )

#     if not reviews_for_llm:
#         logger.info("No usable reviews found for LLM context")
#         return {
#             "answer": "No reviews found for this product.",
#             "reviews": []
#         }
    
#     # --------------------------------------------------
#     # Step 3.5: Rerank reviews using cross-encoder
#     # --------------------------------------------------
#     logger.info(
#         "Starting reranking | candidate_reviews=%d",
#         len(reviews_for_llm)
#     )

#     pairs = [
#         (question, review)
#         for review in reviews_for_llm
#     ]

#     scores = reranker.predict(pairs)

#     scored_reviews = list(zip(reviews_for_llm, scores))

#     # Sort reviews by reranker score (descending)
#     scored_reviews.sort(key=lambda x: x[1], reverse=True)

#     TOP_N = 5
#     MIN_SCORE = 0.02  # optional but recommended

#     reranked_reviews = [
#         review
#         for review, score in scored_reviews
#         if score >= MIN_SCORE
#     ][:TOP_N]

#     # for i, (review, score) in enumerate(scored_reviews[:5], start=1):
#     #     logger.info("RERANK_%d | score=%.4f | text_preview=%s",
#     #             i, score, review[:60])


#     if not reranked_reviews:
#         logger.warning(
#             "Reranker returned 0 results | falling back to top Pinecone reviews"
#         )
#         reranked_reviews = reviews_for_llm[:TOP_N]

#     logger.info(
#         "Reranking completed | selected_reviews=%d",
#         len(reranked_reviews)
#     )

#     # --------------------------------------------------
#     # Step 4: Build context for LLM
#     # --------------------------------------------------
#     context = "\n\n".join(reranked_reviews)
#     logger.info(
#         "Context built for LLM | reviews_used=%d | context_chars=%d",
#         len(reviews_for_llm),
#         len(context)
#     )

#     # --------------------------------------------------
#     # Step 5: Call LLM
#     # --------------------------------------------------
#     logger.info("Calling LLM to generate answer")

#     answer = (
#         prompt
#         | llm
#         | StrOutputParser()
#     ).invoke({
#         "context": context,
#         "question": question
#     }).strip()

#     logger.info("LLM response received")

#     # --------------------------------------------------
#     # Step 6: Validate answer
#     # --------------------------------------------------
#     if not answer or len(answer.split()) < 5:
#         logger.info(
#             "LLM answer weak or empty | returning fallback message"
#         )
#         answer = "Customers have not clearly mentioned this in their reviews."

#     logger.info("CHAT_RESPONSE_READY")

#     reranked_set = set(reranked_reviews)

#     filtered_ui_reviews = [
#         r for r in reviews_for_ui
#         if r["review_text"] in reranked_set
#     ]

#     return {
#         "answer": answer,
#         "reviews": filtered_ui_reviews
#     }

