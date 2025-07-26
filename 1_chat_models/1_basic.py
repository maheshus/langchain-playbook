from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Invoke a model with a query
result = llm.invoke("What is your name?")
print(f"[Result] {result}\n")
print(f"[Content] {result.content}")
