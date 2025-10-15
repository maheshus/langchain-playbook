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
db_dir = os.path.join(current_dir, "db")

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

# --- Function to create and persist vector store ---
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")
        
# Hugging Face Transformers
# Uses models from the Hugging Face library. Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings, https://huggingface.co/models?library=sentence-transformers

logger.info("\n--- Using Hugging Face embeddings (all-MiniLM-L6-v2) ---")
hf_embeddings_minilm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
logger.info("all-MiniLM-L6-v2 embeddings initialized.")
create_vector_store(docs, hf_embeddings_minilm, "chroma_db_hf_minilm")

logger.info("\n--- Using Hugging Face embeddings (all-mpnet-base-v2) ---")
hf_embeddings_mpnet = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
logger.info("all-mpnet-base-v2 embeddings initialized.")
create_vector_store(docs, hf_embeddings_mpnet, "chroma_db_hf_mpnet")

# Function to query a vector store
def query_vector_store(vector_store_name, embedding_fn, query):
    persistent_directory = os.path.join(db_dir, vector_store_name)

    if os.path.exists(persistent_directory):
        # --- Load existing Chroma vector store with embeddings ---
        logger.info(f"Loading vector store from: {persistent_directory}")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_fn)
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
query_vector_store("chroma_db_hf_minilm", hf_embeddings_minilm, query)
query_vector_store("chroma_db_hf_mpnet", hf_embeddings_mpnet, query)
