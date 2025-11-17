import datetime
import logging

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
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


# Catalog of tools exposed to the agent for execution.
tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description=(
            "Retrieves the current system time. "
            "Use this when the agent needs precise, real-time timestamps "
            "for logging, decision-making, or workflow coordination."
        ),
    )
]

# Retrieve the ReAct prompt template (Reason–Action pattern) from the LangChain Hub.
# https://smith.langchain.com/hub/hwchase17/react
logger.info(
    "Fetch the ReAct (Reason + Action) prompt template from the LangChain Hub..."
)
prompt = hub.pull("hwchase17/react")

# Create an agent that uses ReAct prompting
logger.info("Set up the ReAct agent using the create_react_agent function.")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=True)

# Build the agent executor by combining the agent logic and available tools.
logger.info(
    "Create an agent executor that connects the agent with the tools it can use."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

response = agent_executor.invoke({"input": "What is the current time?"})
print("response:", response)
