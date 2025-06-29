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
messages = [
    SystemMessage(
        content="You are a helpful assistant. Respond concisely, professionally and technically."
    ),
    HumanMessage(content="What is LangChain? What is it useful for?"),
]

# Invode a model with messages
result = llm.invoke(messages)
print(f"[AI Message] {result.content}")
