import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
    # To include a timestamp, you need to specify %(asctime)s in the format string when configuring logging
)
logger = logging.getLogger(__name__)

# --- Define paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_w_metadata")

# --- Initialize Hugging Face sentence transformer model ---
# 'all-MiniLM-L6-v2' generates dense vector representations for input text
# These embeddings support similarity-based semantic search via vector stores
logger.info("Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embeddings initialized.")

# --- Load existing Chroma vector store with embeddings ---
logger.info(f"Loading vector store from: {persistent_directory}")
# Chroma supports semantic search and retrieval based on vector similarity
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
logger.info("Vector store loaded successfully.")

# --- Define user query ---
query = "Who was the command module pilot for Apollo 11?"
# query = "Who is Romeo?"
logger.info(f"Query: {query}")

# --- Retrieve relevant documents using similarity score threshold ---
logger.info("Retrieving relevant documents...")

# search_type="similarity_score_threshold" performs a semantic similarity search
# and filters out documents that fall below the specified similarity score
# 'score_threshold' is a float between 0 and 1:
# - Higher values (e.g., 0.8–0.95) return only closely matching documents
# - Lower values (e.g., 0.3–0.5) include broader results
# Useful for controlling the strictness of semantic relevance in retrieval
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
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
