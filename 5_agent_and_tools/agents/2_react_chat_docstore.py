# Standard library imports
import getpass
import logging
import os
from urllib.parse import quote

# Third-party library imports
import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Configure logging, to include a timestamp — you need to specify %(asctime)s in the format string when configuring logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# Get system user name
user = getpass.getuser()

# Load environment variables from .env file
load_dotenv()


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

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", "4_rag", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_w_metadata")

# Initialize Hugging Face sentence transformer model
logger.info(
    "Initializing Hugging Face embeddings (sentence-transformers/all-mpnet-base-v2)..."
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
logger.info("Embeddings initialized.")

# Load existing Chroma vector store with embeddings
if os.path.exists(persistent_directory):
    logger.info(f"Loading vector store from: {persistent_directory}")
else:
    raise FileNotFoundError(
        f"The directory '{persistent_directory}' does not exist. Please check the path."
    )

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
    search_kwargs={"score_threshold": 0.3, "k": 3},
)

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, standalone_qn_pt
)

# This system prompt guides the LLM to generate accurate, well-grounded answers
# strictly using the retrieved context.
# Must contain input variable "context" (override by setting document_variable),
# which will be used for passing in the formatted documents.

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
qa_pt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_sys_pt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

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


def get_wikipedia_summary(query: str) -> str:
    """
    Fetch a concise summary for the given topic using the Wikipedia REST API.

    This function uses the public MediaWiki REST endpoint to retrieve
    a short extract for the requested topic. It requires no API key
    and is fully open-source friendly.

    Args:
        query (str): The topic or keyword to look up on Wikipedia.

    Returns:
        str: A brief summary for the topic. If the topic is not found or an
        error occurs, a clear fallback message is returned instead of raising
        an exception.
    """
    try:
        # Encode the query to make it URL-safe (handles spaces & special chars)
        encoded_query = quote(query)

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
        headers = {"User-Agent": "Mozilla/5.0"}

        # Log that we are making an external Wikipedia API call
        logger.info(f"Calling Wikipedia API for query: '{query}' -> {url}")
        response = requests.get(
            url, headers=headers, timeout=10
        )  # timeout - Maximum seconds to wait for the API response.

        if response.status_code != 200:
            logger.error(
                f"No Wikipedia page found for '{query}'. Status: {response.status_code}"
            )
            return f"I couldn't find any information on '{query}'."

        data = response.json()
        extract = data.get("extract")

        if not extract:
            logger.warning(f"Wikipedia returned no extract for '{query}'.")
            return f"No summary available for '{query}'."

        return extract

    except Exception as e:
        logger.exception(
            f"Unexpected error while searching Wikipedia for '{query}': {e}"
        )
        return "Something went wrong while searching Wikipedia."


# Catalog of tools exposed to the agent for execution.
tools = [
    Tool(
        name="answer_question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description=(
            "Use this tool to answer questions by retrieving and reasoning over "
            "the indexed context using the RAG pipeline."
        ),
    ),
    Tool(
        name="get_wikipedia_summary",
        func=get_wikipedia_summary,
        description=(
            "Fallback tool. Use this ONLY if the RAG tool does not provide sufficient "
            "information. Searches Wikipedia for general public knowledge."
        ),
    ),
]

# Retrieve the ReAct prompt template (Reason–Action pattern) from the LangChain Hub.
logger.info(
    "Fetch the hwchase17/structured-chat-agent prompt template from the LangChain Hub..."
)
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize chat hsitory using InMemoryChatMessageHistory.
chat_history = InMemoryChatMessageHistory()

# create_structured_chat_agent sets up a chat agent that communicates using structured tool calls.
# It links the LLM, the tool definitions, and the prompt template to form an interactive agent.
logger.info("Set up the ReAct agent using the create_structured_chat_agent function.")
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Build the agent executor by combining the agent logic and available tools.
logger.info(
    "Create an agent executor that connects the agent with the tools it can use."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Add an initial system message to establish context for the agent
# SystemMessage defines instructions that guide the agent's behavior throughout the conversation
initial_message = (
    "You are an AI assistant capable of providing helpful answers using the available tools. "
    "If you cannot answer directly, you may rely on the following tools: answer_question, get_wikipedia_summary."
)
chat_history.add_message(SystemMessage(content=initial_message))

# Chat loop to continuously interact with the user
while True:
    user_input = input(f"{user}: ")
    if user_input.lower() == "exit":
        break

    # Store the user's message in chat history
    chat_history.add_user_message(user_input)

    # Run the agent with the user's input and existing chat history
    response = agent_executor.invoke(
        {"input": user_input, "chat_history": chat_history.messages}
    )
    print("AI:", response["output"])

    # Store the agent's reply in chat history
    chat_history.add_ai_message(response["output"])
