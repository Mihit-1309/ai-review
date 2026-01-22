from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sklearn.metrics.pairwise import cosine_similarity

from components.llm import load_llm
from components.embeddings import embed_text

SIMILARITY_THRESHOLD = 0.85


TOPIC_PROMPT = """
You are an assistant that extracts concise semantic topics from customer reviews.

RULES:
- Extract EXACTLY 3 topics
- Each topic must be 1–3 words
- No synonyms or overlapping meanings
- Topics must represent product aspects
- Do NOT repeat similar topics

Return ONLY valid JSON array of strings.

Review:
{review}

Output:
"""


def extract_topics(review: str) -> list[str]:
    llm = load_llm()

    chain = (
        PromptTemplate(
            template=TOPIC_PROMPT,
            input_variables=["review"]
        )
        | llm
        | JsonOutputParser()
    )

    try:
        return chain.invoke({"review": review})
    except Exception:
        return []


def merge_topics(topic_objects: list[dict]) -> list[dict]:
    merged = []

    for obj in topic_objects:
        found = False

        for m in merged:
            similarity = cosine_similarity(
                [obj["embedding"]],
                [m["embedding"]]
            )[0][0]

            if similarity >= SIMILARITY_THRESHOLD:
                m["count"] += 1
                found = True
                break

        if not found:
            merged.append({
                "topic": obj["topic"],
                "embedding": obj["embedding"],
                "count": 1
            })

    return merged


def generate_top_topics(reviews: list[str], top_k: int = 10):
    all_topics = []

    for review in reviews:
        topics = extract_topics(review)

        for t in topics:
            all_topics.append({
                "topic": t,
                "embedding": embed_text(t)
            })

    merged = merge_topics(all_topics)
    merged.sort(key=lambda x: x["count"], reverse=True)

    return [
        {"topic": t["topic"], "count": t["count"]}
        for t in merged[:top_k]
    ]
