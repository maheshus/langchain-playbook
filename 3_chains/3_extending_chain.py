from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
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

# Define additional processing steps using RunnableLambda
convert_to_upper = RunnableLambda(lambda content: content.upper())
count_words = RunnableLambda(lambda content: f"{content}\n\n[Metadata] Words: {len(content.split())} | Characters: {len(content)}") # type: ignore

# Build the full processing chain using LCEL
# Flow: prompt → model → output parser → uppercase → word count
chain = (
    pt | llm | StrOutputParser() | convert_to_upper | count_words
)

# Optional: Keep Post-Processing Modular
# If you plan to reuse the post-processing steps, you could group them:
# postprocess = convert_to_upper | count_words
# chain = prompt_template | llm | StrOutputParser() | postprocess

# Run chain with a sample query
result = chain.invoke({"q": "The Hubble Telescope"})
print(result)
