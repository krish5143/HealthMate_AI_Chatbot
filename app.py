from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sys

# 1. UPDATED IMPORTS (Keep these)
from langchain_groq import ChatGroq
from src.helper import (
    download_hugging_face_embeddings,
)
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Imports from your project structure
from src.prompt import *

# --- Global Initialization ---
app = Flask(__name__)
load_dotenv()

# --- CRITICAL FIX: Initialize the heavy RAG chain at the global scope ---
# This code runs when Gunicorn first loads the 'app' module, before any workers are forked.
# This prevents the first web request from timing out while the model loads.
try:
    # 💥 Call the function directly here to initialize the cache for all workers!
    RAG_CHAIN_INSTANCE = None

    # -------------------------------------------------------------
    # ⚠️ IMPORTANT: We need to adapt your initialization function
    # to be called once outside of a request context.
    # The original function structure is slightly complex for this.
    # Let's run the core logic here instead of inside the function call,
    # as Gunicorn executes this code exactly once per worker process.
    # -------------------------------------------------------------

    # Set up API keys (The OS environment is already set by load_dotenv)
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    if not PINECONE_API_KEY or not GROQ_API_KEY:
        print(
            "ERROR: Required API keys are missing. Check .env file or environment variables.",
            file=sys.stderr,
        )
        # We can't exit here, as Gunicorn needs the 'app' object.

    # 1. Load Embeddings (The heaviest operation, takes time)
    print(
        "--- 🧠 Starting heavy RAG chain initialization... (Runs when worker boots) ---"
    )
    print("Loading embeddings model...")
    embeddings = download_hugging_face_embeddings()

    # 2. Connect to Pinecone Index
    index_name = "healthmate-chatbot"
    print(f"Connecting to Pinecone index: {index_name}...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    # 3. Create Retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 4. Initialize LLM and Chains
    chatModel = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

    # Final RAG Chain
    RAG_CHAIN_INSTANCE = create_retrieval_chain(retriever, question_answer_chain)

    print("--- Initialization successful! RAG chain is ready. (Worker is live) ---")

except Exception as e:
    # If the heavy setup fails, log the error but allow Flask to start
    # so the error message can be displayed by the route.
    print(f"FATAL ERROR during RAG chain initialization: {e}", file=sys.stderr)
    RAG_CHAIN_INSTANCE = None


# --- Function to Handle Heavy Initialization (The original function is now OBSOLETE) ---
# We keep this as a stub just in case you use it elsewhere, but it's no longer needed in /get
def initialize_rag_chain():
    # If this is called, just return the global instance
    return RAG_CHAIN_INSTANCE


# --- Flask Routes ---


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Now, we just check and use the *already loaded* global instance
    if RAG_CHAIN_INSTANCE is None:
        return (
            "Service is not initialized. Check server logs for API key or model loading errors.",
            503,
        )

    rag_chain = RAG_CHAIN_INSTANCE

    msg = request.form.get("msg")
    if not msg:
        return "No message provided", 400

    # ... rest of your code remains the same

    print(f"User Input: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}", file=sys.stderr)
        return (
            "Sorry, the service encountered an internal error. "
            "The model may be overloaded or the API keys may be invalid."
        ), 500


if __name__ == "__main__":
    # Standard development server launch
    app.run(host="0.0.0.0", port=8080, debug=True)
