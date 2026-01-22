from flask import Flask, render_template, request, jsonify
from itertools import islice
from components.topics import generate_top_topics

from components.retriever import create_qa_chain
from common.logger import get_logger
from common.custom_exception import CustomException
from collections import Counter
import re
from components.vector_store import load_vector_store
from components.database import reviews_collection
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

# -------------------------------------------------------------------------------
@app.route("/reviews", methods=["POST"])
def add_review():
    try:
        data = request.get_json(force=True)

        required = ["review_id", "product_id", "product_name", "wsid", "rating", "review_text"]

        for field in required:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400
            
        data["rating"] = int(data["rating"])
        data["embedded"] = False  # so listener embeds it
        reviews_collection.insert_one(data)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to insert review"}), 500
    

@app.route("/reviews/<product_id>", methods=["GET"])
def get_reviews(product_id):
    try:
        wsid = request.args.get("wsid")

        if not wsid:
            return jsonify({"error": "wsid is required"}), 400

        vectorstore = load_vector_store()
        if not vectorstore:
            return jsonify({"error": "Vector store not available"}), 500

        docs = vectorstore.similarity_search(
            query="review",   # neutral query
            k=50,
            filter={
                "WSID": str(wsid),
                "product_id": str(product_id)
            }
        )

        results = []
        for d in docs:
            results.append({
                "review_text": d.page_content,
                "product_id": d.metadata.get("product_id"),
                "product_name": d.metadata.get("product_name"),
                "rating": d.metadata.get("rating"),
                "wsid": d.metadata.get("WSID")
            })

        return jsonify(results)

    except Exception as e:
        logger.error("Failed to fetch reviews from Pinecone", exc_info=True)
        return jsonify([]), 200


# --------------------------------------------------------------------------------
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
        product_id = str(data.get("product_id"))
        question = data.get("question")

        summary_type = data.get("summary_type", "neutral")

        # ✅ STRICT validation
        if not wsid or not product_id:
            return jsonify({"error": "WSID and product_id required"}), 400
        
        if not question:
            return jsonify({"error": "Question is required"}), 400

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
        result = qa_chain.invoke(question)

        response = {
            "answer": "",
            "topics": []
        }
        logger.info(f"RAW LLM RESULT: {result}")
        if summary_type == "neutral":

    # 🔹 If LLM returned STRING (StrOutputParser)
            if isinstance(result, str):
                response["answer"] = result.strip()

            # 🔹 If LLM returned DICT (JsonOutputParser)
            elif isinstance(result, dict) and "summary" in result:
                response["answer"] = result["summary"].strip()

            else:
                response["answer"] = ""

        # =========================
        # POSITIVE / NEGATIVE
        # =========================
        else:
            if isinstance(result, dict) and "topics" in result:
                response["topics"] = [
                    t for t in result["topics"]
                    if t.get("summary") and t["summary"].strip()
                ]

    #             ]
    #     if isinstance(result, dict):

    # # -------- NEUTRAL --------
    #         if "summary" in result:
    #             summary = result.get("summary", "").strip()

    #             # If LLM refused or returned empty → retry once
    #             # if (
    #             #     not summary
    #             #     or "no review" in summary.lower()
    #             #     or "can't generate" in summary.lower()
    #             #     or "insufficient" in summary.lower()
    #             # ):
    #             #     # retry neutral generation once
    #             #     result_retry = qa_chain.invoke("reviews")
    #             #     summary = result_retry.get("summary", "").strip()

    #             response["answer"] = summary

    #         # -------- POSITIVE / NEGATIVE --------
    #         elif "topics" in result:
    #             response["topics"] = [
    #                 t for t in result["topics"]
    #                 if t.get("summary") and t["summary"].strip()
    #         ]


    #     else:
    #         response["answer"] = result

        return jsonify(response)

    except Exception as e:
        logger.error(f"Ask endpoint failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to generate summary"
        }), 500
    

@app.route("/topics", methods=["POST"])
def topics():
    """
    Generate top semantic topics from reviews
    """
    try:
        data = request.get_json(force=True)
        reviews = data.get("reviews")

        if not reviews or not isinstance(reviews, list):
            return jsonify({"error": "reviews must be a list of strings"}), 400

        topics = generate_top_topics(reviews)

        return jsonify({
            "topics": topics
        })

    except Exception as e:
        logger.error("Topic generation failed", exc_info=True)
        return jsonify({
            "error": "Failed to generate topics"
        }), 500

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
