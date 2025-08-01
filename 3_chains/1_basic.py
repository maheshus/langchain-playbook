from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Template with placeholder
pt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Hubble, an expert astrophysicist. Your role is to assist users with questions on physics and astrophysics, tailoring explanations to their level of understanding—from beginners to seasoned experts. Provide clear, insightful, and accurate answers to help deepen their knowledge of the universe.",
        ),
        ("human", "Hey Hubble, can you help me with——{q}?"),
    ]
)

# Build chain using LangChain Expression Language (LCEL)
chain = (
    pt | llm | StrOutputParser()
)  # Define the full processing chain: prompt → model → output parser

# Instead of writing multiple steps—constructing prompts,
# invoking the model, parsing the output—every time, a chain lets you bundle them into a single unit
# You can add logging, memory, agents, tools, or more complex logic without changing your existing structure.
# Chains make this plug-and-play.

# Run chain with a sample query
result = chain.invoke({"q": "The Hubble Telescope"})
print(result)
