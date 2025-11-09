import os
import logging
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Define paths for database persistence
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_quotes")

# Initialize Hugging Face embeddings model
logger.info("Loading Hugging Face model 'all-MiniLM-L6-v2' for embedding generation...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embedding model successfully initialized.")

# Check if vector store already exists
if not os.path.exists(persistent_dir):
    logger.info("No existing Chroma vector store found. Starting new index creation process...")

    # Step 1: Load environment and validate API key
    webpage_url = "https://quotes.toscrape.com/"
    load_dotenv()

    # Step 2: Crawl target site using FireCrawl
    logger.info(f"Initiating content extraction from {webpage_url} using FireCrawl...")
    loader = FireCrawlLoader(url=webpage_url, mode="scrape")
    documents = loader.load()
    logger.info(f"Content extraction completed. Total documents fetched: {len(documents)}")

    # Step 3: Normalize metadata values
    logger.info("Normalizing document metadata fields...")
    for doc in documents:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))
    logger.info("Metadata normalization complete.")

    # Step 4: Split long documents into smaller, retrievable chunks
    logger.info("Performing text segmentation into 1000-character chunks...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    logger.info(f"Segmentation complete. Total chunks generated: {len(chunks)}")
    logger.debug(f"Sample chunk preview:\n{chunks[0].page_content}\n")

    # Step 5: Build and persist Chroma vector store
    logger.info("Creating Chroma vector index from text chunks...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_dir)
    logger.info(f"Chroma index successfully created and stored at: {persistent_dir}")

else:
    logger.info("Existing Chroma vector store detected. Skipping index creation step.")

# Step 6: Load the existing Chroma vector store
logger.info(f"Loading Chroma vector store from persistent directory: {persistent_dir}")
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
logger.info("Chroma vector store successfully loaded and ready for queries.")

# Step 7: Initialize retriever interface for semantic search
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant results
)
logger.info("Retriever initialized with similarity search (top_k=3).")

# Step 8: Execute semantic query
query = "Quotes by Albert Einstein."
logger.info(f"Executing semantic search for query: '{query}'")
relevant_docs = retriever.invoke(query)
logger.info(f"Query execution completed. Retrieved {len(relevant_docs)} matching chunks.")

# Step 9: Display results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, start=1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
