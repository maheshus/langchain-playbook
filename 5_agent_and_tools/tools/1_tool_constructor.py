import logging
from math import cos, radians, sqrt

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool, Tool
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


def get_iss_location(_=None) -> dict:
    """
    Fetch the real-time ISS position using the WhereTheISS.at API.

    Returns:
        dict with ISS latitude, longitude, altitude, velocity, and footprint.
    """
    url = "https://api.wheretheiss.at/v1/satellites/25544"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


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


def can_see_iss(user_lat: float, user_lon: float) -> dict:
    """
    Estimate whether the ISS may be visible from a user’s position.

    Uses a simple flat-earth approximation:
      - Fetch current ISS subsatellite point (lat/lon)
      - Compute ground distance to user
      - Compare with ISS footprint radius (~visibility circle)

    Returns:
        dict containing ground distance, visibility radius, and a human-readable explanation.
    """
    iss = get_iss_location()  # <-- automatic internal call
    iss_lat = iss["latitude"]
    iss_lon = iss["longitude"]
    footprint = iss["footprint"]  # diameter in km
    radius_km = footprint / 2

    # Simple flat-earth distance
    KM_LAT = 111  # km per degree latitude
    KM_LON = 111 * cos(radians(user_lat))  # km per degree longitude

    dx = (iss_lon - user_lon) * KM_LON
    dy = (iss_lat - user_lat) * KM_LAT

    distance = sqrt(dx * dx + dy * dy)

    visible = distance <= radius_km

    return {
        "distance_km": round(distance, 1),
        "visibility_radius_km": round(radius_km, 1),
        "visible": visible,
        "explanation": (
            f"Distance to ISS ground point is {distance:.0f} km. "
            f"Visibility radius is {radius_km:.0f} km. "
            + ("You may see it!" if visible else "Too far away.")
        ),
    }


# Pydantic Schemas Used for Structured Tools
class ReverseGeocodeArgs(BaseModel):
    lat: float = Field(description="Latitude of the location")
    lon: float = Field(description="Longitude of the location")


class CanSeeISSArgs(BaseModel):
    user_lat: float = Field(description="User latitude")
    user_lon: float = Field(description="User longitude")


# Catalog of tools exposed to the agent for execution.
tools = [
    # 1. Simple tool, no args → uses Tool class
    Tool(
        name="GetISSLocation",
        func=get_iss_location,
        description=(
            "Use this tool whenever the user asks about the ISS location, "
            "This tool returns the real-time latitude, longitude, altitude, and velocity of the ISS."
        ),
    ),
    # 2. Structured tool with multiple inputs
    StructuredTool.from_function(
        func=reverse_geocode,
        name="ReverseGeocode",
        description=(
            "Use this tool when the user provides latitude and longitude and wants to know "
            "which place it corresponds to. It returns country code, timezone, and map URL."
        ),
        args_schema=ReverseGeocodeArgs,  # type:ignore
    ),
    # 3. Structured tool with multiple inputs
    StructuredTool.from_function(
        func=can_see_iss,
        name="CanSeeISS",
        description=(
            "Determine if the ISS may be visible from the user's location. "
            "This tool ONLY requires user_lat and user_lon. "
            "It automatically fetches the current ISS latitude, longitude, and altitude."
        ),
        args_schema=CanSeeISSArgs,
    ),
]

# Retrieve the ReAct prompt template (Reason–Action pattern) from the LangChain Hub.
logger.info("Fetch the openai-tools-agent prompt template from the LangChain Hub...")
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create an agent that uses ReAct prompting
logger.info("Set up the tool calling agent.")
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Build the agent executor by combining the agent logic and available tools.
logger.info(
    "Create an agent executor that connects the agent with the tools it can use."
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

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

# Example 4 — Another combined query
response = agent_executor.invoke(
    {"input": "Can I see the ISS if I'm at latitude 19.076 and longitude 72.8777?"}
)
print("\nResponse 4 (Visibility reasoning):", response)
