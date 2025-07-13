from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
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

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: pt.format_prompt(**x))
call_llm = RunnableLambda(lambda x: llm.invoke(x.to_messages()))  # type: ignore
extract_content = RunnableLambda(lambda x: x.content)

# Build chain using RunnableSequence [ equivalent to LangChain Expression Language (LCEL)]
# Compose the full pipeline: user input → format_prompt → call_llm → extract_content
# `middle` is a list to allow multiple intermediate steps
chain = RunnableSequence(first=format_prompt, middle=[call_llm], last=extract_content)

# Note: LCEL (using `|`) is preferred for simpler chains—more concise and readable.
# Equivalent chain using LCEL (LangChain Expression Language) is written: chain = format_prompt | call_llm | extract_content
# RunnableSequence is useful when building dynamically or when explicit control is needed.

# Run chain with a sample query
result = chain.invoke({"q": "QZSS comparison to other navigation systems like GPS."})
print(result)
