import datetime
import logging
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
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


# Create a basic tool function that fetches the current time.
def get_current_time(*args, **kwargs) -> str:
    """
    Return the current local system time in 12-hour format (HH:MM AM/PM).

    Args:
        *args: Ignored. Present for compatibility with agent/tool interfaces.
        **kwargs: Ignored. Present for compatibility with agent/tool interfaces.

    Returns:
        str: Formatted current time (e.g., "03:45 PM").
    """
    return datetime.datetime.now().strftime("%I:%M %p")


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
        name="get_current_time",
        func=get_current_time,
        description=(
            "Returns the current system time in HH:MM AM/PM format. "
            "Use this for time-based reasoning or scheduling."
        ),
    ),
    Tool(
        name="get_wikipedia_summary",
        func=get_wikipedia_summary,
        description=(
            "Searches Wikipedia for the given topic and returns a short summary. "
            "Useful for general facts, background information, and quick knowledge lookup."
        ),
    ),
]

# Retrieve the ReAct prompt template (Reason–Action pattern) from the LangChain Hub.
logger.info(
    "Fetch the hwchase17/structured-chat-agent prompt template from the LangChain Hub..."
)
prompt = hub.pull("hwchase17/structured-chat-agent")

# Initialize a structured chat agent using ConversationBufferMemory.
# The memory component preserves the full dialogue history, enabling the agent
# to understand and reference past interactions.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create_structured_chat_agent sets up a chat agent that communicates using structured tool calls.
# It links the LLM, the tool definitions, and the prompt template to form an interactive agent.
logger.info("Set up the ReAct agent using the create_structured_chat_agent function.")
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Build the agent executor by combining the agent logic and available tools.
logger.info(
    "Create an agent executor that connects the agent with the tools it can use."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)

# Add an initial system message to establish context for the agent
# SystemMessage defines instructions that guide the agent's behavior throughout the conversation
initial_message = (
    "You are an AI assistant capable of providing helpful answers using the available tools. "
    "If you cannot answer directly, you may rely on the following tools: Time and Wikipedia."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop to continuously interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Store the user's message in memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Run the agent with the user's input and existing chat history
    response = agent_executor.invoke({"input": user_input})
    print("AI:", response["output"])

    # Store the agent's reply in memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
