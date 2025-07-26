import os
import logging

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
file_path = os.path.join(current_dir, "books", "rockets_missiles_and_spacecraft.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# --- Check if vector store already exists ---
if not os.path.exists(persistent_directory):
    logger.info("Vector store not found. Creating a new one...")

    # --- Validate text file existence ---
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # --- Load text content ---
    logger.info("Loading document...")
    txt_loader = TextLoader(file_path, autodetect_encoding=True)
    raw_docs = txt_loader.load()

    # --- Split text into chunks ---
    logger.info("Splitting document into chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(raw_docs)
    logger.info(f"Document split into {len(docs)} chunks.")
    logger.info(f"Sample chunk:\n{docs[0].page_content}\n")

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
