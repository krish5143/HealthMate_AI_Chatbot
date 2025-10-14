from flask import Flask, render_template, jsonify, request

# Import ChatGroq for fast LLM inference
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

# Set up API keys: Pinecone (for the vector DB) and Groq (for the LLM)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Switched from OPENAI_API_KEY

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY  # Switched from OPENAI_API_KEY

# Assuming download_hugging_face_embeddings correctly returns the embedding model
embeddings = download_hugging_face_embeddings()

# --- Your confirmed index name ---
index_name = "healthmate-chatbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the ChatGroq model (llama-3.3 is fast and powerful)
chatModel = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        # Assuming system_prompt in src.prompt is structured to include {context}
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

# Note: For the RAG chain to work correctly, the system_prompt *must* be structured to accept {context}
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        # Better error handling for API failures
        print(f"Error during RAG chain invocation: {e}")
        return (
            "Sorry, I encountered an internal error while processing your request. "
            "Please check your GROQ_API_KEY and connection."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
