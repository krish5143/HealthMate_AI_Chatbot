from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sys

# 1. UPDATED IMPORTS for LangChain 0.2+ (Addressing warnings/future errors)
from langchain_groq import ChatGroq
from src.helper import (
    download_hugging_face_embeddings,
)  # Assuming this is updated in helper.py
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Imports from your project structure
from src.prompt import *

# --- Global Initialization (Only lightweight setup here) ---
app = Flask(__name__)
load_dotenv()

# Placeholder for the fully initialized RAG chain (heavy object)
# This will hold the result of the initialization function
rag_chain_cache = None


# --- Function to Handle Heavy Initialization (The Fix for 502/Timeout) ---
def initialize_rag_chain():
    global rag_chain_cache

    # Check if the cache is already populated on a previous request
    if rag_chain_cache is not None:
        return rag_chain_cache

    print("--- 🧠 Starting heavy RAG chain initialization... (Runs only once) ---")

    try:
        # Set up API keys (Good practice to set these early)
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

        if not PINECONE_API_KEY or not GROQ_API_KEY:
            raise ValueError(
                "Required API keys (PINECONE_API_KEY or GROQ_API_KEY) are missing."
            )

        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

        # 1. Load Embeddings (The heaviest operation, takes time)
        print("Loading embeddings model...")
        embeddings = download_hugging_face_embeddings()

        # 2. Connect to Pinecone Index
        index_name = "healthmate-chatbot"
        print(f"Connecting to Pinecone index: {index_name}...")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name, embedding=embeddings
        )

        # 3. Create Retriever
        retriever = docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

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
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Cache the successful result
        rag_chain_cache = rag_chain
        print("--- Initialization successful! RAG chain is ready. ---")
        return rag_chain_cache

    except Exception as e:
        print(f"FATAL ERROR during RAG chain initialization: {e}", file=sys.stderr)
        # If initialization fails, prevent the app from serving requests
        # In a real app, you might want a placeholder or more sophisticated retry logic.
        sys.exit(1)  # Force the worker process to exit if initialization fails


# --- Flask Routes ---


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Retrieve the initialized chain. This calls the function on the first request only.
    rag_chain = initialize_rag_chain()

    msg = request.form.get("msg")
    if not msg:
        return "No message provided", 400

    print(f"User Input: {msg}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}", file=sys.stderr)
        # Give the user a clear indication of a server issue
        return (
            "Sorry, the service encountered an internal error. "
            "The model may be overloaded or the API keys may be invalid."
        ), 500


if __name__ == "__main__":
    # Note: When deploying with Gunicorn on Render, this block is usually ignored.
    # The Gunicorn command in Render's configuration is what matters!
    # Ensure your Gunicorn Start Command is: gunicorn --bind 0.0.0.0:$PORT app:app
    app.run(host="0.0.0.0", port=8080, debug=True)
