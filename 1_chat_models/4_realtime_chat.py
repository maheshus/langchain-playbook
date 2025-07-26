from dotenv import load_dotenv
import getpass
from utils.llm_utils import get_model_name
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Get the local logged-in username
user = getpass.getuser()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Use a list track chat history
chat_history = []

# Use Case: Enterprise IT Helpdesk Assistant (AI Ops)
# Set an initial system message (optional)
system_message = SystemMessage(
    content="You are a helpful enterprise IT helpdesk assistant (AI Ops)"
)
# Add SystemMessage to chat history
chat_history.append(system_message)

# Chat loop
while True:
    q = input(f"{user}: ")
    if q.lower() == "exit":
        break

    # Add HumanMessage to chat history
    chat_history.append(HumanMessage(content=q))

    # Get response from llm using history
    response = llm.invoke(chat_history)

    # Add AIMessage to chat history
    chat_history.append(AIMessage(content=response.content))

    print(f"{get_model_name(llm)}: {response.content}")

# Diplay chat history on exit
print("---- Message History ----")
print(chat_history)
