# https://python.langchain.com/docs/integrations/chat/
from dotenv import load_dotenv
from utils.llm_utils import get_model_name
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# ---- LangChain OpenAI Chat Model Example ----
# LangChain's ChatOpenAI is built to work with OpenAI's API by default.
# However, since OpenRouter uses the same API structure (endpoints, headers, JSON format),
# we can point ChatOpenAI to OpenRouter by setting:
#   - `OPENAI_API_BASE` to OpenRouter's API URL
#   - `OPENAI_API_BASE` to your OpenRouter API key
# This allows us to use OpenRouter-hosted models (e.g., Mistral, LLaMA, Claude)
# without writing a custom wrapper — everything works through the standard OpenAI interface.
llm_deepseek = ChatOpenAI(model="deepseek/deepseek-chat-v3-0324:free")

# ---- Google Chat Model Example ----
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

# Use case involving a medical appointment assistant
messages = [
    SystemMessage(
        content="You are a healthcare assistant that helps book doctor appointments, gives health tips, and answers general medical questions."
    ),
    HumanMessage(
        content="Hi, I think I need to see a doctor for a sore throat. Can you help me book an appointment?"
    ),
    AIMessage(
        content="Of course! May I know your location so I can find nearby clinics?"
    ),
    HumanMessage(content="I'm in San Francisco."),
]

# Invoke models with messages
result = llm_deepseek.invoke(messages)
print(f"[AI Message][{get_model_name(llm_deepseek)}] {result.content}")

result = llm_gemini.invoke(messages)
print(f"[AI Message][{get_model_name(llm_gemini)}] {result.content}")
