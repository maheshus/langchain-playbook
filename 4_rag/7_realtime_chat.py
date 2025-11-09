# Standard library imports
import getpass
import logging
import os
import time

# Third-party library imports
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# Configure logging, to include a timestamp — you need to specify %(asctime)s in the format string when configuring logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Get system user name
user = getpass.getuser()

# Load environment variables from .env file
load_dotenv()

# region Model Selection
def select_llm():
    """Prompt user to select between Gemini and OpenRouter models."""
    print(f"\n{user}, please select a model:")
    print("1. Google Gemini (gemini-2.0-flash)")
    print("2. Google Gemini (gemini-2.0-flash-lite)")
    print("3. OpenRouter (mistralai/mistral-nemo:free)")

    choice = input("Enter your choice (1, 2 or 3): ").strip()
    match choice:
        case "1":
            print("✓ Using Google gemini-2.0-flash.\n")
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        case "2":
            print("✓ Using Google gemini-2.0-flash-lite.\n")
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
        case "3":
            print("✓ Using OpenRouter mistralai/mistral-nemo.\n")
            return ChatOpenAI(model="mistralai/mistral-nemo:free")
        case _:
            print("Invalid choice, defaulting to Google gemini-2.0-flash-lite.")
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# Initialize selected LLM
llm = select_llm()
#endregion

# region Vector Store Creation
# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_w_metadata")

# Initialize Hugging Face sentence transformer model
logger.info("Initializing Hugging Face embeddings (sentence-transformers/all-mpnet-base-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
logger.info("Embeddings initialized.")

# Load existing Chroma vector store with embeddings
if os.path.exists(persistent_directory):
    logger.info(f"Loading vector store from: {persistent_directory}")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    logger.info("Vector store loaded successfully.")

# search_type="similarity_score_threshold" performs a semantic similarity search
# and filters out documents that fall below the specified similarity score
# 'score_threshold' is a float between 0 and 1:
# - Higher values (e.g., 0.8–0.95) return only closely matching documents
# - Lower values (e.g., 0.3–0.5) include broader results
# Useful for controlling the strictness of semantic relevance in retrieval
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4, "k": 5},
)


# endregion

# region History-Aware Retriever
# Contextual question prompt
# Instructs the AI to rewrite the user’s question using the chat history
# so the query becomes self-contained and understandable on its own

standalone_qn_sys_prompt = """You are given a conversation history and the user's most recent question.

Your goal is to restate the latest question as a clear, self-contained version that fully captures
the user's intent — including any context, entities, or relationships mentioned earlier in the chat.

If the user's question uses pronouns or vague references (like "he," "she," "they," or "this person"),
replace them with the correct names or details inferred from the conversation history.

Do NOT answer the question — only rewrite it. If the question is already self-contained, return it unchanged."""

# Build a chat prompt template to reformulate user questions based on context
# This prompt helps the model take the conversation history and the latest user query,
# and rewrite it into a self-contained question that doesn't rely on prior messages.
# Useful in RAG pipelines where the retriever expects a standalone query for better accuracy.

standalone_qn_pt = ChatPromptTemplate.from_messages(
    [
        ("system", standalone_qn_sys_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This retriever uses the LLM to rewrite the user's question by incorporating relevant
# details from the chat history before performing document retrieval.
# It ensures that follow-up questions are contextualized into complete, standalone queries,
# improving the accuracy of RAG results.

history_aware_retriever = create_history_aware_retriever(llm, retriever, standalone_qn_pt)
# endregion

# region Question Answering Prompt
# This system prompt guides the LLM to generate accurate, well-grounded answers 
# strictly using the retrieved context. If the information is missing or uncertain, 
# the model should clearly state that it doesn't know — avoiding guesses or assumptions.
# Responses should be clear, factual, and limited to three concise sentences.
qa_sys_pt = """You are an expert assistant answering questions based solely on the retrieved context below.
The context may include different sections of a literary work.

Use the information to answer clearly, even if the story uses different words (e.g., “poisoned” means “died by poison”).
If the answer is not explicitly stated but can be safely inferred from context, explain it briefly.

If there is no relevant information, respond with:
"I'm not sure based on the available information."

Keep responses factual and within five sentences.

Context:
{context}"""

# Build a prompt template for answering user questions
# Combines the system instructions, chat history, and current user input
# to generate grounded, context-aware responses.
qa_pt = ChatPromptTemplate.from_messages([
    ("system", qa_sys_pt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ]
)
# endregion

# region Retrieval-Augmented Generation Pipeline
# Build a chain for question answering using retrieved documents.
# `create_stuff_documents_chain` builds a chain that sends all retrieved documents 
# (the "stuffed" context) directly into the LLM prompt at once.  
# It’s useful for smaller sets of documents where the entire context can fit within 
# the model’s token limit, enabling the LLM to generate an informed and coherent answer.
qa_chain = create_stuff_documents_chain(llm, qa_pt)

# Combine the history-aware retriever with the QA chain.
# `create_retrieval_chain` links a retriever with a question-answering (QA) chain.
# It first retrieves relevant documents using the retriever, 
# then passes them to the QA chain, which generates an answer based on that context.
# This creates a complete Retrieval-Augmented Generation (RAG) pipeline in a single step.
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
#endregion

# Function to simulate an ongoing interactive chat session with the AI
def run_chat_session():
    """Interactive chat session with fixed delay between LLM requests."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []

    while True:
        user_input = input(f"{user}: ")
        if user_input.lower() == "exit":
            print("Session ended. Goodbye!")
            break

        try:
            # Send query to the RAG chain
            result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            ai_answer = result.get("answer", "No response generated!")
            print(f"AI: {ai_answer}")

            # Update chat history
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=ai_answer)
            ])

            # Delay to prevent hitting rate limits
            time.sleep(3)

        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(3)  # small delay before retrying next user input


# Entry point of the script
# Runs the run chat session loop when the script is executed directly
if __name__ == "__main__":
    run_chat_session()
