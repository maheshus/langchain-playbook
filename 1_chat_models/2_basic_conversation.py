from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# SystemMessage: Configures AI behavior (first in chain)
# - Role/persona setup | Response format | Safety rules | Objectives

# HumanMessage: User input to AI
# - Questions | Commands | Context | Multimedia

# AIMessage: Generated response
# - Answers | Computations | Structured data | Follow-ups

# This is a conversation without an AIMessage
messages = [
    SystemMessage(
        content="Your are a travel assistant that books flights, finds hotels, and answers questions"
    ),
    HumanMessage(
        content="I want to go to Paris next weekend. Can you help me book a flight?"
    ),
]

# Invoke a model with messages
result = llm.invoke(messages)
print(f"[AI Message] {result.content}\n")

# A full-blown conversation, this enables contextual awareness
# Use case involving a travel assistant
messages = [
    SystemMessage(
        content="Your are a travel assistant that books flights, finds hotels, and answers questions"
    ),
    HumanMessage(
        content="I want to go to Paris next weekend. Can you help me book a flight?"
    ),
    AIMessage(
        content="Sure! Let me check flights from your nearest airport. Where are you flying from?"
    ),
    HumanMessage(content="I'm in New York."),
    AIMessage(
        content="Here are some options: Delta: Friday 6 PM, $450; Air France: Saturday 9 AM, $500. Would you like to book one of these?"
    ),
    HumanMessage(content="The Delta one looks good. Book it."),
]

# Invoke a model with messages
result = llm.invoke(messages)
print(f"[AI Message] {result.content}")
