from flask import Flask, render_template, request, jsonify, session
from itertools import islice
from components.topics.processor import process_new_reviews
from components.database import topic_store
from components.retriever import create_qa_chain
from common.logger import get_logger
from common.custom_exception import CustomException
from collections import Counter
import re
from components.vector_store import load_vector_store
from components.database import reviews_collection
logger = get_logger(__name__)
from components.chatbot.chain import chat_with_reviews
from flask import session
import uuid


app = Flask(__name__)
app.secret_key = "dev-secret-key-123"



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

  

        return jsonify(response)

    except Exception as e:
        logger.error(f"Ask endpoint failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to generate summary"
        }), 500
    


@app.route("/topics/top", methods=["POST"])
def get_top_topics():
    try:
        data = request.get_json(force=True)
        WSID = data.get("WSID")
        product_id = data.get("product_id")

        print("DEBUG: WSID =", WSID)
        print("DEBUG: product_id =", product_id)

        if not WSID or not product_id:
            raise ValueError("WSID or product_id missing")

        from components.topics.processor import process_new_reviews
        process_new_reviews(WSID, product_id)

        from components.database import topic_store
        topics = list(
            topic_store.find(
                {"wsid": WSID, "product_id": product_id},
                {"_id": 0, "topic": 1, "count": 1}
            )
            .sort("count", -1)
            .limit(10)
        )

        print("DEBUG: topics found =", topics)

        return jsonify({"topics": topics})

    except Exception as e:
        import traceback
        print("\n🔥🔥🔥 ERROR IN /topics/top 🔥🔥🔥")
        traceback.print_exc()
        print("🔥🔥🔥 END ERROR 🔥🔥🔥\n")

        return jsonify({
            "error": str(e)
        }), 500
    
    
     
@app.route("/api/reviews-by-topic", methods=["GET"])
def get_reviews_by_topic():
    topic = request.args.get("topic")
    wsid = request.args.get("wsid")
    product_id = request.args.get("product_id")

    if not topic or not wsid or not product_id:
        return jsonify({"error": "Missing required params (topic, wsid, product_id)"}), 400

    topic_doc = topic_store.find_one({
        "topic": topic,
        "wsid": wsid,
        "product_id": product_id
    })

    if not topic_doc:
        return jsonify({"reviews": []}), 200

    review_ids = topic_doc.get("review_ids", [])
    
    if not review_ids:
        return jsonify({"reviews": []}), 200
    

    reviews = list(reviews_collection.find({
        "review_id": {"$in": review_ids}
    }, {"_id": 0}))  # Hide internal Mongo _id

    return jsonify({"reviews": reviews})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)

    wsid = data.get("wsid")
    product_id = data.get("product_id")
    question = data.get("question")

    if not wsid or not product_id or not question:
        return jsonify({"error": "Missing parameters"}), 400

    # -------------------------------------
    # Create session ID (per browser session)
    # -------------------------------------

    if "chat_history" not in session:
        # session["chat_id"] = str(uuid.uuid4())
        session["chat_history"] = []

    chat_history = session["chat_history"]

    # Call chatbot
    response = chat_with_reviews(
        wsid=wsid,
        product_id=product_id,
        question=question,
        chat_history=chat_history
    )

    # Store conversation
    chat_history.append({
        "role": "user",
        "content": question
    })
    chat_history.append({
        "role": "assistant",
        "content": response["answer"]
    })

    session["chat_history"] = chat_history

    return jsonify(response)


@app.route("/reset-session", methods=["POST"])
def reset_session():
    session.clear()
    return jsonify({"status": "session cleared"})



# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
