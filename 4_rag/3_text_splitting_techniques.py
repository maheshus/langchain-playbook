import os
import logging

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
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
db_dir = os.path.join(current_dir, "db")

# --- Validate text file existence ---
if not os.path.exists(file_path):
    logger.error(f"Input file not found: {file_path}")
    raise FileNotFoundError(f"Input file not found: {file_path}")

# --- Load text content ---
logger.info("Loading document...")
txt_loader = TextLoader(file_path, autodetect_encoding=True)
raw_docs = txt_loader.load()

# --- Create embeddings using a Hugging Face sentence-transformer model ---
logger.info("Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embeddings initialized.")


# Function to create and persist vector store
def create_vector_store(docs, vector_store_name):
    persistent_directory = os.path.join(db_dir, vector_store_name)
    if not os.path.exists(persistent_directory):
        logger.info(f"Vector store not found. Creating: {vector_store_name}...")
        # --- Create and persist the vector store ---
        logger.info("Creating and persisting Chroma vector store...")

        # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.as_retriever
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        logger.info(f"Vector store created and saved at: {persistent_directory}")
    else:
        logger.info(f"Vector store {vector_store_name} already exists. Skipping creation.")


# --- 1. Character-based Splitting ---
# Splits text into fixed-size chunks (by characters).
# Useful for predictable chunk sizes, regardless of content structure.
print("\n--- Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(raw_docs)
create_vector_store(char_docs, "chroma_db_char")

# --- 2. Sentence-based Splitting ---
# Splits text at sentence boundaries, keeping chunks semantically coherent.
# Useful when you want full sentences instead of mid-sentence splits.
print("\n--- Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(raw_docs)
create_vector_store(sent_docs, "chroma_db_sent")

# --- 3. Token-based Splitting ---
# Splits text into chunks based on tokens (subwords/words).
# Essential when using transformer models with strict token limits.
print("\n--- Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(raw_docs)
create_vector_store(token_docs, "chroma_db_token")

# --- 4. Recursive Character-based Splitting ---
# Attempts to split text at natural boundaries (paragraphs, sentences).
# Balances readability with size constraints → often best for RAG pipelines.
print("\n--- Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(raw_docs)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# --- 5. Custom Splitting ---
# Define your own logic for splitting.
# Example: here we split by paragraphs (two consecutive newlines).
print("\n--- Custom Splitting ---")


class CustomTextSplitter(TextSplitter):
    def split_text(self, text: str):
        return text.split("\n\n")  # Example: split by paragraphs


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(raw_docs)
create_vector_store(custom_docs, "chroma_db_custom")


# Function to query a vector store
def query_vector_store(vector_store_name, query):
    persistent_directory = os.path.join(db_dir, vector_store_name)

    if os.path.exists(persistent_directory):
        # --- Load existing Chroma vector store with embeddings ---
        logger.info(f"Loading vector store from: {persistent_directory}")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        logger.info("Vector store loaded successfully.")

        logger.info("Retrieving relevant documents...")
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.3},
        )
        relevant_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(relevant_docs)} relevant document(s).")

        # --- Display results ---
        print("\n--- Relevant Documents ---")
        # Loop through each doc in relevant_docs, and also keep track of its position starting from 1 (not 0)
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\nDocument {i}:\n{doc.page_content}")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
    else:
        logger.error(f"Vector store {vector_store_name} is invalid!")


# Define the user's question
query = "Who was the command module pilot for Apollo 11?"

# Query each vector store
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
