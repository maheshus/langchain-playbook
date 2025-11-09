import os
import logging

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging, to include a timestamp, you need to specify %(asctime)s in the format string when configuring logging
logging.basicConfig( level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# File and directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_3iatlas")

# Create embeddings using a Hugging Face sentence-transformer model ---
# This initializes 'all-MiniLM-L6-v2', a lightweight transformer that converts text chunks into fixed-size vectors
# These embeddings are used for semantic search in LangChain-compatible vector stores (e.g., Chroma, FAISS)
logger.info("Initializing Hugging Face embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embeddings initialized.")

# Check if vector store already exists
if not os.path.exists(persistent_directory):
    logger.info("Vector store not found. Creating a new one...")

    # Step 1: Fetch content from the target webpage
    # WebBaseLoader retrieves and parses webpage text content, making it ready for embedding or retrieval tasks.
    urls = ["https://en.wikipedia.org/wiki/3I/ATLAS"]

    # Step 2: Create a loader to fetch and parse web content
    # WebBaseLoader downloads the target pages and extracts clean text data for downstream processing.
    logger.info("Loading data from web...")
    loader = WebBaseLoader(urls, autoset_encoding=True)
    documents = loader.load()

    # Step 3: Split the scraped content into manageable chunks
    # CharacterTextSplitter divides long text into smaller, structured segments
    # to improve embedding accuracy and retrieval efficiency.
    logger.info("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Document split into {len(docs)} chunks.")
    logger.info(f"Sample chunk:\n{docs[0].page_content}\n")

    # Step 5: Create and persist the vector store
    logger.info("Creating and persisting Chroma vector store...")
    # Chroma supports semantic search and retrieval based on vector similarity
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    logger.info(f"Vector store created and saved at: {persistent_directory}")

else:
    logger.info("Vector store already exists. Skipping creation.")
    
# Load existing Chroma vector store with embeddings ---
logger.info(f"Loading vector store from: {persistent_directory}")
# Chroma supports semantic search and retrieval based on vector similarity
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
logger.info("Vector store loaded successfully.")

# Step 6: Query the vector store for relevant information
# The retriever interfaces with the vector database to find semantically similar chunks.
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # Retrieve the top 3 most relevant results
)

# Define the user's question
query = "What is 3I/ATLAS? Is it dangerous?"

# Retrieve the most relevant document chunks based on semantic similarity
relevant_docs = retriever.invoke(query)

# Display the retrieved documents along with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, start=1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
