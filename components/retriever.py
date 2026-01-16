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

        # Retrieve top 10 documents
        # retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
        filter_dict = {
            # "WSID": wsid,
            "product_id": f"{product_id}.0"
        }

        if summary_type == "positive":
            filter_dict["rating"] = {"$gte": 4}

        elif summary_type == "negative":
            filter_dict["rating"] = {"$lte": 3}

        retriever = vectorstore.as_retriever(
          search_kwargs={
              "k": 100,
              "filter": filter_dict
              })
          



        if summary_type == "positive":
            selected_prompt = POSITIVE_PROMPT
            parser = JsonOutputParser()
        elif summary_type == "negative":
            selected_prompt = NEGATIVE_PROMPT
            parser = JsonOutputParser()
        else:
            selected_prompt = NEUTRAL_PROMPT
            parser = JsonOutputParser()

        prompt = PromptTemplate(
            template=selected_prompt,
            # input_variables=["context"]
            input_variables=["context","product_name"]
        )

        def retrieve_context(_):
            docs = retriever.invoke("reviews")

            logger.info(f"Retrieved {len(docs)} docs for summary")

            if not docs:
                return {
                    "context": "",
                    "product_name": ""
                }

            product_name = docs[0].metadata.get("product_name", "This product")
            context = "\n\n".join(d.page_content for d in docs)

            return {
                "context": context,
                "product_name": product_name
            }






    #     def retrieve_context(_):
    #         docs = retriever.invoke("reviews")

    #         logger.info(f"Retrieved {len(docs)} docs for summary")

    #         if not docs:
    #             # 🚨 IMPORTANT: short-circuit with valid JSON
    #             if summary_type in ["positive", "negative"]:
    #                 return {
    #                     "context": "",
    #                     "product_name": ""
    #                 }
    #             else:
    #                 return {
    #                     "context": "",
    #                     "product_name": "This product"
    #                 }

    #         product_name = docs[0].metadata.get("product_name", "This product")
    #         context = "\n\n".join(d.page_content for d in docs)

    #         return {
    #             "context": context,
    #             "product_name": product_name
    # }

        def guard_empty_context(inputs):
            if not inputs["context"].strip():
                # ✅ Return valid JSON directly
                if summary_type == "positive":
                    return {"topics": []}
                elif summary_type == "negative":
                    return {"topics": []}
                else:
                    return {"summary": "No reviews available for this product."}
            return inputs
        

        docs = retriever.invoke("reviews")
        logger.info(f"Retrieved {len(docs)} docs for summary")
        if not docs:
            logger.info("No reviews found, skipping LLM")

            def empty_response(_):
                if summary_type in ["positive", "negative"]:
                    return {"topics": []}
                else:
                    return {"summary": "No reviews available for this product."}

            return RunnableLambda(empty_response)

        # 🔹 BUILD CONTEXT FROM METADATA
        product_name = docs[0].metadata.get("product_name", "This product")
        context = "\n\n".join(
            f"{d.metadata.get('review_title', '')} {d.metadata.get('review_text', '')}"
            for d in docs
        )

#         chain = (
#           RunnableLambda(retrieve_context)|
#           RunnableLambda(guard_empty_context)
#           | prompt
#           | llm
#           | parser
# )
        def format_docs(docs):
            reviews = []

            for d in docs:
                if isinstance(d.metadata, dict):
                    reviews.append(d.metadata.get("review_text", ""))

            product_name = (
                docs[0].metadata.get("product_name", "This product")
                if docs else "This product"
            )

            return {
                "context": "\n\n".join(reviews),
                "product_name": product_name
            }

        chain = (
            retriever
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