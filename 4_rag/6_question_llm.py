import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
    # To include a timestamp, you need to specify %(asctime)s in the format string when configuring logging
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_w_metadata")

# Initialize Hugging Face sentence transformer model
logger.info(
    "Initializing Hugging Face embeddings (sentence-transformers/all-mpnet-base-v2)..."
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
logger.info("Embeddings initialized.")


def query_vector_store(store_name, query, embedding_fn, srch_type, srch_kwargs):
    # Load existing Chroma vector store with embeddings
    if os.path.exists(persistent_directory):
        logger.info(f"Loading vector store from: {persistent_directory}")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_fn)
        logger.info("Vector store loaded successfully.")
        retriever = db.as_retriever(
            search_type=srch_type,
            search_kwargs=srch_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(relevant_docs)} relevant document(s).")

        return relevant_docs
    else:
        logger.warning(f"Chroma vector store {store_name} does not exist.")
        return None


# Define user query
# query = "What are the main types of spacecraft mentioned?"
query = "How did Juliet die?"
logger.info(f"Query: {query}")

logger.info("Using Similarity Score Threshold ")
relevant_docs = query_vector_store(
    "chroma_db_with_metadata",
    query,
    embeddings,
    "similarity_score_threshold", {"k": 4, "score_threshold": 0.4},
)

# Combine relevant document contents into a clear, structured prompt for the LLM.
context = (
    "Here are reference documents you can use:\n\n"
    + "\n\n".join([f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}" for i, doc in enumerate(relevant_docs)])  # type: ignore
    + "\n\nPlease answer using only the information above."
)

system_prompt = (
    "You are Luna, an intelligent and trustworthy librarian assistant. "
    "Your task is to answer questions based strictly on the reference documents provided. "
    "Use the context to enrich and support your responses with accurate, relevant details. "
    "Always prioritize information directly drawn from the provided context — do not guess or fabricate details. "
    "If the context does not contain enough information to answer confidently, respond with: 'I'm not sure.' "
    "When possible, cite your sources naturally (e.g., mention the book title, section, or author) to make your response informative and credible. "
    "Focus on clarity, factual accuracy, and helpfulness while maintaining a calm, professional tone."
)

pt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Question: {query}"),
        ("human", "{context}"),
    ]
)

p = pt.invoke({"query": query, "context": context})
result = llm.invoke(p)

# Display the full result and content only
print("\nGenerated Response")
# print("Full result:")
# print(result)
print(f"AI: {result.content}")
