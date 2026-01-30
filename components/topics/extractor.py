from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from components.llm import load_llm

TOPIC_PROMPT = """
Extract EXACTLY 3 concise product topics from the review.

Rules:
- 1–3 words per topic
- Product aspects only
- No sentiment
- No repetition
- No explanation

Return ONLY a JSON array.

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

def extract_topics(review_text: str) -> list[str]:
    try:
        topics = chain.invoke({"review": review_text})
        if isinstance(topics, list) and len(topics) == 3:
            return topics
        return []
    except Exception:
        return []
