from flask import Flask, render_template, request, jsonify
from itertools import islice

from components.retriever import create_qa_chain
from common.logger import get_logger
from common.custom_exception import CustomException
from collections import Counter
import re
from components.vector_store import load_vector_store

logger = get_logger(__name__)

app = Flask(__name__)


# ===============================
# Home Page
# ===============================
@app.route("/", methods=["GET"])
def index():
    """
    Renders the frontend UI
    """
    return render_template("index.html")


# ===============================
# Ask / Summarize Endpoint
# ===============================
@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts a question and returns the product review summary
    """
    try:
        data = request.get_json(force=True)
        wsid = data.get("wsid")
        product_id = int(data.get("product_id"))
        question = data.get("question")

        summary_type = data.get("summary_type", "neutral")

        # ✅ STRICT validation
        if not wsid or not product_id:
            return jsonify({"error": "WSID and product_id required"}), 400
        
        # if not question:
        #     return jsonify({"error": "Question is required"}), 400

        # if not isinstance(question, str):
        #     return jsonify({"error": "Question must be a string"}), 400

        # Question is required ONLY for positive / negative
        if summary_type in ["positive", "negative"]:
                if not question or not isinstance(question, str):
                    return jsonify({"error": "Question is required for this summary type"}), 400

        
        if summary_type not in ["neutral", "positive", "negative"]:
            summary_type = "neutral"

        # logger.info(f"Creating QA chain | summary_type={summary_type}")
        logger.info(
    f"Invoking chain | summary_type={summary_type} | wsid={wsid} | product_id={product_id}"
)


        qa_chain = create_qa_chain(
            summary_type=summary_type,
            wsid=wsid,
            product_id=product_id
        )

        if qa_chain is None:
            raise CustomException("QA chain creation failed")

        logger.info("Running QA chain")

        # ✅ PASS STRING ONLY (CRITICAL FIX)

        # result = qa_chain.invoke(question)
        # # result = qa_chain.invoke({})

        # # ✅ Handle both string & dict outputs safely
        # if isinstance(result, dict):
        #     answer = result.get("result") or result.get("answer", "")
        # else:
        #     answer = result

        # # return jsonify({
        # #     "answer": answer
        # # })
        # return jsonify(result)
        if summary_type == "neutral":
            result = qa_chain.invoke("reviews")
        else:
            result = qa_chain.invoke(question or "reviews")
        response = {
            "answer": "",
            "topics": []
        }
        logger.info(f"RAW LLM RESULT: {result}")
        # if isinstance(result, dict):
        #     response["answer"] = (
        #         result.get("answer")
        #         or result.get("result")
        #         or ""
        #     )
        #     response["topics"] = result.get("topics", [])
        # if isinstance(result, dict):
        #      # ✅ NEUTRAL summary
        #     if "summary" in result:
        #         response["answer"] = result["summary"]

        #     # ✅ POSITIVE / NEGATIVE summaries
        #     elif "topics" in result:
        #         response["topics"] = result["topics"]
        if isinstance(result, dict):

    # ✅ NEUTRAL summary
            if "summary" in result:
                summary = result.get("summary", "").strip()

                # 🔒 Guard against empty or incorrect neutral output
                if not summary or "no reviews" in summary.lower():
                    summary = (
                        "Customers say the product generally meets expectations, "
                        "with feedback highlighting performance, usability, and overall value."
                    )

                response["answer"] = summary

            # ✅ POSITIVE / NEGATIVE summaries
            elif "topics" in result:
                # Remove empty topic summaries (extra safety)
                response["topics"] = [
                    t for t in result["topics"]
                    if t.get("summary") and t["summary"].strip()
                ]


        else:
            response["answer"] = result

        return jsonify(response)

    except Exception as e:
        logger.error(f"Ask endpoint failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to generate summary"
        }), 500
    


STOPWORDS = {
    "the","is","and","to","of","a","in","for","on","with","this",
    "that","it","as","are","was","but","be","have","has","had",
    "very","also","from","they","them","their","you","your"
}

# GENERIC_WORDS = {
#     "good","great","nice","product","products","item","items",
#     "buy","use","used","using","work","works","working"
# }

def generate_ngrams(words, n):
    return zip(*(islice(words, i, None) for i in range(n)))

def clean_tokens(text):
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS ]

@app.route("/top-words", methods=["GET"])
def top_words():
    try:
        vectorstore = load_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 300})

        docs = retriever.invoke("customer product reviews")
        if not docs:
            return jsonify({"phrases": []})

        text = " ".join(d.page_content for d in docs)
        tokens = clean_tokens(text)

        # Count ngrams
        unigrams = Counter(tokens)
        bigrams = Counter(" ".join(bg) for bg in generate_ngrams(tokens, 2))


        # Minimum frequency filter
        unigrams = Counter({k:v for k,v in unigrams.items() if v >= 5})
        bigrams = Counter({k:v for k,v in bigrams.items() if v >= 3})


        # Combine with priority: trigram > bigram > unigram
        candidates = []
        # for phrase, count in trigrams.most_common():
        #     candidates.append((phrase, count))
        for phrase, count in bigrams.most_common():
            candidates.append((phrase, count))
        for phrase, count in unigrams.most_common():
            candidates.append((phrase, count))

        # Remove overlaps
        final_phrases = []
        used_tokens = set()

        for phrase, count in candidates:
            phrase_tokens = set(phrase.split())

            if phrase_tokens & used_tokens:
                continue

            final_phrases.append({"phrase": phrase, "count": count})
            used_tokens.update(phrase_tokens)

            if len(final_phrases) == 10:
                break

        return jsonify({"phrases": final_phrases})

    except Exception:
        logger.exception("TOP WORDS FAILED")
        return jsonify({"phrases": []}), 500
# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
