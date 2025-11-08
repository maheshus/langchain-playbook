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
logger.info("Initializing Hugging Face embeddings (sentence-transformers/all-mpnet-base-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
logger.info("Embeddings initialized.")

def query_vector_store(store_name, query, embedding_fn, srch_type, srch_kwargs):
    # --- Load existing Chroma vector store with embeddings ---
    if os.path.exists(persistent_directory):
        logger.info(f"Loading vector store from: {persistent_directory}")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_fn)
        logger.info("Vector store loaded successfully.")

        retriever = db.as_retriever(search_type=srch_type, search_kwargs=srch_kwargs,)
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
        logger.warning(f"Chroma vector store {store_name} does not exist.")

# --- Define user query ---
# query = "Who was the command module pilot for Apollo 11?"
query = "How did Juliet die?"
logger.info(f"Query: {query}")

# Different retrieval methods
# 1. Similarity Search
# Retrieves the top-k documents most similar to a given query using cosine similarity.
# Useful for finding contextually relevant content in the vector store.
logger.info("--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity", {"k": 3})

# 2. Max Marginal Relevance (MMR)
# Retrieves documents using Max Marginal Relevance (MMR), balancing relevance and diversity.
# 'fetch_k' defines how many top similar documents to consider before re-ranking.
# 'lambda_mult' adjusts the trade-off:
#   - 1 → more relevant, less diverse
#   - 0 → more diverse, less focused
# Use MMR to avoid redundant results and ensure a broader, more informative document set.
logger.info("--- Max Marginal Relevance (MMR) ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

# 3. Similarity Score Threshold
# Retrieves documents based on a minimum similarity score threshold.  
# Only documents with a similarity score above 'score_threshold' are returned.  
# Useful for filtering out weak matches and keeping only highly relevant results.  
# Ensures quality over quantity in retrieved documents.  
logger.info("--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity_score_threshold", {"k": 3, "score_threshold":0.3})

logger.info("Querying demonstrations with different search types completed.")