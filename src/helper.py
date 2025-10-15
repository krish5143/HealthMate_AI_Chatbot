# Fixed helper.py with modern LangChain imports

# 1. UPDATED IMPORTS from langchain-community and langchain-text-splitters
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List
from langchain.schema import Document
# If you are using langchain-core for Document, use this instead:
# from langchain_core.documents import Document


# Extract Data From the PDF File
def load_pdf_file(data):
    """Loads all PDF files from the specified directory."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.

    This function is unchanged as it seems correct for filtering metadata,
    but ensures it works with the correct Document class if you swap it.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        # Note: The 'source' is usually in 'metadata' when loaded by DirectoryLoader
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data):
    """Splits the documents into text chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    """
    Downloads the specified HuggingFace embedding model.
    Using HuggingFaceEmbeddings from langchain_community.
    """
    # This is a key heavy operation that causes the deployment delay.
    # It must be called ONLY when needed (as fixed in app.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Adding a cache directory is often useful for repeated deployment/local testing
        # model_kwargs={'device': 'cpu'},
    )
    return embeddings
