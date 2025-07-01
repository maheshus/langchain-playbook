# Example Source: https://python.langchain.com/docs/integrations/memory/google_firestore/
import os
import getpass
from dotenv import load_dotenv
from utils.llm_utils import get_model_name
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

# Load environment variables from .env file
load_dotenv()

# Get the local logged-in username
user = getpass.getuser()

# Initialize Firestore client
print("Initialize Firestore Client...")
filestore_client = firestore.Client(project=os.getenv("FIRESTORE_PROJECT_ID"))

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=str(os.getenv("FIRESTORE_SESSION_ID")),
    collection=str(
        os.getenv("FIRESTORE_COLLECTION", "langchain_session")
    ),  # Default to langchain_session if not set
    client=filestore_client,
)

print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

print(
    "Start chatting with the AI.\nType 'exit' to quit or 'clear' to reset the chat history.\n"
)
while True:
    user_input = input(f"{user}: ")
    if user_input.lower() == "exit":
        break

    if user_input.lower() == "clear":
        chat_history.clear()
        print("Chat history cleared!")
        continue

    chat_history.add_user_message(user_input)
    result = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(result.content)  # type: ignore

    print(f"{get_model_name(llm)}: {result.content}\n")
