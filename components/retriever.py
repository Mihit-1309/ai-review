from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from components.llm import load_llm
from components.vector_store import load_vector_store
from langchain_core.runnables import RunnableLambda
from common.logger import get_logger
from common.custom_exception import CustomException

logger = get_logger(__name__)


NEUTRAL_PROMPT = """
You are an assistant that MUST generate a neutral product summary
based on customer reviews.

IMPORTANT RULES (STRICT):
- Reviews ARE PROVIDED in the context
- You MUST generate a summary
- You are NOT allowed to say that reviews are missing
- You are NOT allowed to say information is insufficient
- Do NOT refuse the task

TASK:
- Read the customer reviews
- Identify recurring points such as quality, performance, value, compatibility, or usability
- Combine them into ONE concise paragraph
- Use balanced, factual language

Style:
- Start with "Customers say..."
- Be neutral and informative

Context:
{context}

Return ONLY valid JSON in this format:
{{
  "summary": "Customers say ..."
}}
"""


POSITIVE_PROMPT = """


---------------
You are an AI assistant that analyzes customer reviews and produces
topic-based POSITIVE summaries.

RULES:
- Use ONLY the provided reviews
- Focus ONLY on positive feedback
- Ignore all negative opinions
- Generate EXACTLY 4 distinct topics
- Topics must be meaningful product aspects
- Do NOT repeat similar topics

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "topics": [
    {{
      "topic": "Topic Name",
      "summary": "Customers say {product_name} ..."
    }}
  ]
}}


Context:
{context}
"""

NEGATIVE_PROMPT = """


-----------
You are an AI assistant that analyzes customer reviews and produces
topic-based NEGATIVE summaries.

RULES:
- Use ONLY the provided reviews
- Focus ONLY on complaints and issues
- Ignore all positive feedback
- Generate EXACTLY 4 distinct topics
- Do NOT exaggerate problems

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "topics": [
    {{
      "topic": "Topic Name",
      "summary": "Customers say {product_name} ..."
    }}
  ]
}}


Context:
{context}
"""

# prompt = PromptTemplate(
#     template=CUSTOM_PROMPT_TEMPLATE,
#     input_variables=["context"]
# )



def create_qa_chain(summary_type, wsid, product_id):
    try:
        logger.info("Loading vector store")
        vectorstore = load_vector_store()

        if vectorstore is None:
            raise CustomException("Vector store not loaded")

        logger.info("Loading LLM")
        llm = load_llm()





        if summary_type == "positive":
            selected_prompt = POSITIVE_PROMPT
            parser = JsonOutputParser()
        elif summary_type == "negative":
            selected_prompt = NEGATIVE_PROMPT
            parser = JsonOutputParser()
        else:
            selected_prompt = NEUTRAL_PROMPT
            parser = JsonOutputParser()

        if summary_type == "neutral":
            prompt = PromptTemplate(
                template=selected_prompt,
                input_variables=["context"]
            )
        else:
            prompt = PromptTemplate(
                template=selected_prompt,
                input_variables=["context", "product_name"]
            )



        def format_docs(docs):
            reviews = []

            if docs:
                for d in docs:
                    if d.page_content:
                        reviews.append(d.page_content)
                    elif isinstance(d.metadata, dict):
                        reviews.append(d.metadata.get("review_text", ""))

                product_name = docs[0].metadata.get("product_name", "This product")
            else:
                product_name = "This product"

            context_text = "\n\n".join(reviews).strip()

            if not context_text:
                context_text = "Customers shared mixed feedback across multiple aspects."

            return {
                "context": context_text,
                "product_name": product_name
            }

        


        def fetch_reviews(user_query: str):

            """
            1. Metadata filter (WSID + product_id)
            2. Semantic ranking using user query
            """

            if not user_query:
                # fallback query if user does not type anything
                user_query = "customer review"

            filter_dict = {
                "WSID": str(wsid),          # ✅ EXACT key
                "product_id": str(product_id)  # ✅ EXACT value
            }
            # if summary_type == "positive":
            #     filter_dict["rating"] = {"$gte": 4}
            # elif summary_type == "negative":
            #     filter_dict["rating"] = {"$lte": 3}

            docs = vectorstore.similarity_search(
                query=user_query,   # non-empty query is safer
                k=20,
                filter=filter_dict
            )

            logger.info(
                f"Retrieved {len(docs)} docs for WSID={wsid}, product_id={product_id}"
            )
            logger.info(
            f"Retrieved {len(docs)} docs after metadata + semantic filtering"
           )

            if docs:
                logger.info(f"Sample matched metadata: {docs[0].metadata}")

            return docs



        chain = (
            RunnableLambda(lambda user_query: fetch_reviews(user_query))
            | RunnableLambda(format_docs)
            | prompt
            | llm
            | parser
        )



        logger.info("Runnable chain created successfully")
        
        # Wrap the chain in the adapter so it works with your existing app.py code
        return chain

    except Exception as e:
        logger.error("Failed to create runnable chain", exc_info=True)
        return None