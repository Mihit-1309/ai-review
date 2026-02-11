from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from components.llm import load_llm


TOPIC_PROMPT = """
Extract concise product topics from the review.

Rules:
- Generate 2 or 3 topics ONLY
- Use 2 topics if the review is short or discusses few aspects
- Use 3 topics if the review discusses multiple aspects
- 1–3 words per topic
- Product aspects only
- No sentiment
- No repetition
- No explanation

Return ONLY a JSON array of strings.

Review:
{review}
"""


llm = load_llm()

chain = (
    PromptTemplate(
        template=TOPIC_PROMPT,
        input_variables=["review"]
    )
    | llm
    | JsonOutputParser()
)

# def extract_topics(review_text: str) -> list[str]:
#     try:
#         result = chain.invoke({"review": review_text})

#         # ✅ Case 1: LLM returned a list
#         if isinstance(result, list):
#             return result[:3]

#         # ✅ Case 2: LLM returned an object
#         if isinstance(result, dict):
#             topics = result.get("topics")
#             if isinstance(topics, list):
#                 return topics[:3]

#         return []

#     except Exception as e:
#         print("Topic extraction error:", e)
#         return []
def extract_topics(review_text: str) -> list[str]:
    try:
        result = chain.invoke({"review": review_text})

        if isinstance(result, list):
            # accept 2 or 3 topics only
            if 2 <= len(result) <= 3:
                return result

        if isinstance(result, dict):
            topics = result.get("topics")
            if isinstance(topics, list) and 2 <= len(topics) <= 3:
                return topics

        return []

    except Exception as e:
        print("Topic extraction error:", e)
        return []

