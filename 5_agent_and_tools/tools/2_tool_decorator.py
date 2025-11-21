import logging

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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


# Pydantic Schemas Used for Structured Tools
class ISSLocationArgs(BaseModel):
    """No arguments needed for this tool."""

    pass


class ReverseGeocodeArgs(BaseModel):
    lat: float = Field(description="Latitude of the location")
    lon: float = Field(description="Longitude of the location")


@tool(args_schema=ISSLocationArgs)
def get_iss_location() -> dict:
    """
    Fetch the real-time ISS position using the WhereTheISS.at API.

    Returns:
        dict with ISS latitude, longitude, altitude, velocity, and footprint.
    """
    url = "https://api.wheretheiss.at/v1/satellites/25544"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


@tool(args_schema=ReverseGeocodeArgs)
def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Fetch location metadata for a given latitude/longitude.

    Returns:
        dict with timezone, country code, map URL, and other place details.
    """
    url = f"https://api.wheretheiss.at/v1/coordinates/{lat},{lon}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


# Catalog of tools exposed to the agent for execution.
tools = [get_iss_location, reverse_geocode]

# Retrieve the ReAct prompt template (Reason–Action pattern) from the LangChain Hub.
# https://smith.langchain.com/hub/hwchase17/react
logger.info("Fetch the openai-tools-agent prompt template from the LangChain Hub...")
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create an agent that uses ReAct prompting
logger.info("Set up the tool caling agent.")
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Build the agent executor by combining the agent logic and available tools.
logger.info(
    "Create an agent executor that connects the agent with the tools it can use."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

print("\n-- TOOL METADATA AS SEEN BY THE AGENT --")
for t in tools:
    print("\nName:", t.name)
    print("Description:", t.description)
    print("Args Schema:", t.args_schema)
    print("JSON Schema:", t.args)
    print("-" * 60)

# Example Queries
# Example 1 — Ask for current ISS location
response = agent_executor.invoke({"input": "Where is the ISS right now?"})
print("\nResponse 1 (ISS Location):", response)

# Example 2 — Reverse-geocode coordinates (place details)
response = agent_executor.invoke(
    {"input": "Tell me about the place at latitude 34.05 and longitude -118.24"}
)
print("\nResponse 2 (Reverse Geocode):", response)

# Example 3 — Combining both tools in a natural-language question
response = agent_executor.invoke(
    {"input": "Is the ISS anywhere near 34.05, -118.24 right now?"}
)
print("\nResponse 3 (Combined reasoning):", response)
