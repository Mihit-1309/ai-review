import os
from dotenv import load_dotenv
from pinecone import Pinecone
from components.web_fallback import get_website_content
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from components.embeddings import embed_text
from components.llm import load_llm
from common.logger import get_logger
from langchain_community.chat_message_histories import ChatMessageHistory
import re
from components.database import reviews_collection

logger = get_logger(__name__)
# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

EMBEDDING_DIM = 384  # must match Pinecone index

llm = load_llm()

# -----------------------------
# Global session memory store
# -----------------------------
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# --------------------------------------------------
# Rewrite Prompt (Conversational Retrieval)
# --------------------------------------------------

REWRITE_PROMPT = """
You are an assistant that rewrites follow-up questions into standalone questions.

Your job is ONLY to resolve ambiguous references such as:
- it
- that
- those
- these
- which one
- the first
- the second

If the latest question is already clear and standalone,
return it unchanged.

Do NOT:
- Add extra descriptions
- Add assumptions
- Add new context
- Reinterpret the product
- Modify the meaning

Only clarify ambiguous references if necessary.

Conversation history:
{history}

Latest question:
{question}

Return ONLY the rewritten standalone question.
"""


rewrite_prompt = PromptTemplate(
    template=REWRITE_PROMPT,
    input_variables=["history", "question"]
)

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# --------------------------------------------------
# Prompt
# --------------------------------------------------

CHAT_PROMPT = """
You are an intelligent product assistant.

You are given two sources:

1) Customer Reviews (primary source — opinions, real experiences, satisfaction, complaints)
2) Official Website Information (secondary source — factual specs like price, warranty, page yield, compatibility, OEM numbers)

PRIORITY RULES:

1. Customer Reviews are ALWAYS the primary source.
   - For performance, quality, durability, satisfaction, complaints:
     → Use ONLY Customer Reviews.
     → Ignore website specs unless explicitly requested.

2. Website Information should be used ONLY IF:
   - The question asks about price, warranty, page yield, compatibility, OEM numbers, shipping, or specifications.
   - OR the reviews clearly do not contain the required information.

3. Do NOT mix website specifications into review-based answers unless the question explicitly requires it.

4. If both are relevant (e.g., "Is it worth buying at this price?"):
   → Start with review-based sentiment.
   → Then optionally reference website price.

5. If negative_percentage is provided and the question is about negative aspects,
mention the percentage in your answer to provide context.
If the percentage is very low (e.g., under 5%), clarify that negative experiences are relatively rare.

Only include information directly relevant to the question.
Do not add unrelated specifications.

RESPONSE FORMAT:
- One clear paragraph.
- No bullet points.
- Professional tone.

Previous conversation:
{history}

Negative Review Percentage:
{negative_percentage}

Customer Reviews:
{reviews_context}

Website Information:
{website_context}

User Question:
{question}

Final Answer:
"""



prompt = PromptTemplate(
    template=CHAT_PROMPT,
    input_variables=["history","negative_percentage","reviews_context","website_context", "question"]
)

parser = StrOutputParser()

BASE_PRODUCT_URL = "https://www.swiftink.com/product/"


def generate_product_url(product_name: str):
    """
    Convert product_name to website slug URL.
    """

    logger.info("Generating product URL from product_name=%s", product_name)

    slug = product_name.lower().strip()
    slug = slug.replace(" ", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    product_url = f"{BASE_PRODUCT_URL}{slug}/"

    logger.info("Generated product URL=%s", product_url)

    return product_url

def compute_negative_percentage(product_id: str):
    """
    Compute negative review percentage using rating <= 2.
    """

    total_reviews = reviews_collection.count_documents({
        "product_id": str(product_id)
    })

    negative_reviews = reviews_collection.count_documents({
        "product_id": str(product_id),
        "rating": {"$lte": 2}
    })

    if total_reviews == 0:
        return 0.0

    negative_percentage = round(
        (negative_reviews / total_reviews) * 100,
        2
    )

    logger.info(
        "Negative stats | total=%d | negative=%d | percentage=%.2f%%",
        total_reviews,
        negative_reviews,
        negative_percentage
    )

    return negative_percentage


def is_negative_question(question: str):
    q = question.lower()
    keywords = [
        "negative", "worst", "complaint",
        "drawback", "bad", "issue",
        "problem", "dislike"
    ]
    return any(k in q for k in keywords)

def chat_with_reviews(wsid: str, product_id: str, question: str,chat_history: list = None):



    logger.info(
        "CHAT_REQUEST | wsid=%s | product_id=%s | question=%s",
        wsid, product_id, question
    )

    # --------------------------------------------------
    # Build conversation history text
    # --------------------------------------------------

    history_text = ""

    if chat_history:
        for msg in chat_history[-4:]:  # limit last 8 messages
            history_text += f"{msg['role']}: {msg['content']}\n"

    # --------------------------------------------------
    # Rewrite question using history (if exists)
    # --------------------------------------------------

    standalone_question = question

    if history_text.strip():
        standalone_question = rewrite_chain.invoke({
            "history": history_text,
            "question": question
        }).strip()

    logger.info(f"Standalone question: {standalone_question}")

    # --------------------------------------------------
    # Embed rewritten question
    # --------------------------------------------------

    query_embedding = embed_text(question)
    logger.info("Embedding generated using ORIGINAL user question")
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

    top_score = res.matches[0].score if res.matches else 0.0

    logger.info(
        "Pinecone query completed | fetched_docs=%d | top_score=%.4f",
        len(res.matches),
        top_score
    )


    # --------------------------------------------------
    # Step 3: Extract reviews
    # --------------------------------------------------
    reviews_for_llm = []
    reviews_for_ui = []

    product_name = None
    for i, m in enumerate(res.matches, start=1):
        review_text = m.metadata.get("review_text") or m.metadata.get("text")
        rating = m.metadata.get("rating")
        
            

        if not product_name:
            product_name = m.metadata.get("product_name")
            logger.info("Product name extracted from metadata: %s", product_name)

        logger.info(
            "DOC_%d | score=%.4f | has_text=%s",
            i,
            (m.score or 0.0),
            bool(review_text)
        )
        if not product_name:
            logger.warning("No product_name found in Pinecone metadata")
        
        if review_text:
            reviews_for_llm.append(review_text)
            reviews_for_ui.append({
                "review_text": review_text,
                "rating": rating,
                "product_name": product_name
            })

    if product_name:
        logger.info("Product name extracted from metadata: %s", product_name)
    else:
        logger.warning("No product_name found in Pinecone metadata")

    logger.info(
        "Review extraction completed | usable_reviews=%d",
        len(reviews_for_llm)
    )


    reviews_context = "\n\n".join(reviews_for_llm) if reviews_for_llm else ""

    logger.info(
        "Reviews context prepared | reviews_used=%d | chars=%d",
        len(reviews_for_llm),
        len(reviews_context)
    )
    website_context = ""

    if product_name:
        logger.info("Fetching website content for product: %s", product_name)

        product_url = generate_product_url(product_name)
        website_context = get_website_content(product_url)

        if website_context:
            logger.info(
                "Website content ready | url=%s | chars=%d",
                product_url,
                len(website_context)
            )
        else:
            logger.warning("Website content empty for url=%s", product_url)
    else:
        logger.warning("Product name missing. Website context skipped.")

    # --------------------------------------------------
    # Step 4: Build context for LLM
    # --------------------------------------------------
    # context = "\n\n".join(reviews_for_llm)

    logger.info(
        "Context built for LLM | reviews_used=%d | context_chars=%d",
        len(reviews_for_llm),
        len(reviews_context)
    )
    
    negative_percentage = None

    if is_negative_question(standalone_question):
        negative_percentage = compute_negative_percentage(product_id)
        logger.info("Negative percentage injected: %.2f%%", negative_percentage)
        
    # --------------------------------------------------
    # Step 5: Call LLM
    # --------------------------------------------------
    logger.info("Calling LLM using combined Reviews + Website context")

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "history": history_text,
        "reviews_context": reviews_context,
        "website_context": website_context,
        "question": standalone_question,
        "negative_percentage": negative_percentage
    }).strip()

    logger.info("LLM response received")

    # --------------------------------------------------
    # Step 6: Validate answer
    # --------------------------------------------------
    # if not answer or len(answer.split()) < 5:
    #     logger.info(
    #         "LLM answer weak or empty | returning fallback message"
    #     )
    #     answer = "Customers have not clearly mentioned this in their reviews."

    # --------------------------------------------------
    # Step 6: Website Fallback Logic
    # --------------------------------------------------

    # fallback_message = "Customers have not clearly mentioned this in their reviews."

    # if not answer or len(answer.split()) < 5 or fallback_message in answer:

    #     logger.info("Answer not found in reviews. Triggering website fallback.")

    #     if product_name:
    #         product_url = generate_product_url(product_name)

    #         website_content = get_website_content(product_url)
       

    #         if website_content:

    #             logger.info(
    #                 "Calling LLM using website content | url=%s | chars=%d",
    #                 product_url,
    #                 len(website_content)
    #             )
    #             answer = (
    #                 prompt
    #                 | llm
    #                 | StrOutputParser()
    #             ).invoke({
    #                 "history": history_text,
    #                 "context": website_content,
    #                 "question": standalone_question
    #             }).strip()

    #             logger.info("Website-based LLM response generated successfully")

    #         else:
    #             logger.warning("Website content empty. Returning fallback message.")
    #             answer = fallback_message

    #     else:
    #         logger.warning("Product name missing. Cannot generate product URL.")
    #         answer = fallback_message



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

