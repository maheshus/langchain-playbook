from dotenv import load_dotenv
import getpass
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.llm_utils import get_model_name

# Load environment (e.g., for API key)
load_dotenv()

# Setup
user = getpass.getuser()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
chat_history = []

# Template with placeholder
pt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful enterprise IT helpdesk assistant named Hubble."),
    ("human", "Hey Hubble, can you help me with——{issue}?"),
])

# Start chat
print("Chat with Hubble (type 'exit' to quit)\n")
issue = input(f"{user}, what issue are you facing?\n")

# First turn using template
pt = pt.invoke({"issue": issue})
# Add initial messages to history
chat_history.extend(pt.to_messages())
response = llm.invoke(chat_history)
chat_history.append(AIMessage(content=response.content))
print(f"\n{get_model_name(llm)}: {response.content}\n")

# Chat Loop
while True:
    q = input(f"{user}: ")
    if q.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=q))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"\n{get_model_name(llm)}: {response.content}\n")

# Show message history
print("---- Chat History ----")
print(chat_history)

# Each item in chat_history is a LangChain message object (e.g., HumanMessage, AIMessage)
# These have a .type (role: "human", "ai", "system") and .content (the text)
# This loop prints the role and content of each message in a readable format

# for m in chat_history:
#     print(f"{m.type.capitalize()}: {m.content}")
