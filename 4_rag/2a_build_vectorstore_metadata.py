import os
import logging
import re

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
    # To include a timestamp, you need to specify %(asctime)s in the format string when configuring logging
)
logger = logging.getLogger(__name__)

# --- File and directory paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
logger.info(f"Books directory: {books_dir}")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_w_metadata")

def extract_gutenberg_metadata(text: str) -> dict:
    """Extracts metadata such as title and author from Project Gutenberg eBook text."""
    metadata = {}

    # Find a line that starts with 'Title:' — no matter where it is in the string.
    title_match = re.search(r"^Title:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
    author_match = re.search(r"^Author:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
    language_match = re.search(r"^Language:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)

    if title_match:
        metadata["title"] = title_match.group(1).strip()
    if author_match:
        metadata["author"] = author_match.group(1).strip()
    if language_match:
        metadata["language"] = language_match.group(1).strip()

    return metadata

# --- Check if vector store already exists ---
if not os.path.exists(persistent_directory):
    logger.info("Vector store not found. Creating a new one...")

    # --- Validate text file existence ---
    if not os.path.exists(books_dir):
        logger.error(f"Input folder not found: {books_dir}")
        raise FileNotFoundError(f"Input folder not found: {books_dir}")
    
    # --- List all the books in text format from the books directory ---
    book_files = [file for file in os.listdir(books_dir) if file.endswith(".txt")]

    # --- Load text content ---
    raw_docs_w_metadata = []
    for book_file in book_files:
        logger.info(f"Loading document {book_file}...")
        file_path = os.path.join(books_dir, book_file)
        txt_loader = TextLoader(file_path, autodetect_encoding=True)
        raw_docs = txt_loader.load()

         # Use loaded document content for metadata extraction
        gutenberg_header_snippet = raw_docs[0].page_content[:1000] if raw_docs else ""
        book_metadata = extract_gutenberg_metadata(gutenberg_header_snippet)
        book_metadata["source"] = book_file

        for doc in raw_docs:
            # Add metadata to each document indicating its source
            doc.metadata = book_metadata
            raw_docs_w_metadata.append(doc)

    # --- Split text into chunks ---
    logger.info("Splitting document into chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(raw_docs_w_metadata)
    logger.info(f"Document split into {len(docs)} chunks.")

    # --- Create embeddings using a Hugging Face sentence-transformer model ---
    # This initializes 'all-MiniLM-L6-v2', a lightweight transformer that converts text chunks into fixed-size vectors
    # These embeddings are used for semantic search in LangChain-compatible vector stores (e.g., Chroma, FAISS)
    logger.info("Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embeddings initialized.")

    # --- Create and persist the vector store ---
    logger.info("Creating and persisting Chroma vector store...")
    # Chroma supports semantic search and retrieval based on vector similarity
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    logger.info(f"Vector store created and saved at: {persistent_directory}")

else:
    logger.info("Vector store already exists. Skipping creation.")
