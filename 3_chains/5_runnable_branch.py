from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI llm
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define prompt templates for different feedback types
pts = {
        "classify": ChatPromptTemplate.from_messages([
            ("system", "You are a professional sentiment classifier."),
            ("human", "Classify the sentiment of this customer feedback into one of four categories: positive, negative, neutral, or escalate.\n\nFeedback: {feedback}\n\nRespond with only one word: Positive, Negative, Neutral, or Escalate.")
        ]),

        "positive": ChatPromptTemplate.from_messages([
            ("system", "You are a customer support assistant who writes professional thank-you notes."),
            ("human", "Write a warm and professional thank-you message to the customer for the following positive feedback. Keep it concise and appreciative.\n\nFeedback: {feedback}")
        ]),

        "negative": ChatPromptTemplate.from_messages([
            ("system", "You are a customer support agent who responds to complaints empathetically and professionally."),
            ("human", "Write a calm and empathetic message addressing this negative feedback. Acknowledge the issue, apologize, and offer help or next steps.\n\nFeedback: {feedback}")
        ]),

        "neutral": ChatPromptTemplate.from_messages([
            ("system", "You are a customer support assistant gathering more information from unclear or neutral feedback."),
            ("human", "Politely ask the customer for more details to better understand their neutral feedback. Keep the tone professional and helpful.\n\nFeedback: {feedback}")
        ]),

        "escalate": ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant helping escalate critical feedback to a human agent."),
            ("human", "Write a brief, professional message informing the customer that their feedback has been escalated to a human support specialist for review. Be respectful and ensure the customer feels heard.\n\nFeedback: {feedback}")
        ]),
    }

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (lambda x: "positive" in x, pts["positive"] | llm | StrOutputParser()), # type: ignore
    (lambda x: "negative" in x, pts["negative"] | llm | StrOutputParser()), # type: ignore
    (lambda x: "neutral" in x, pts["neutral"] | llm | StrOutputParser()),   # type: ignore
    pts["escalate"] | llm | StrOutputParser() # Fallback
)

# Create the classification chain
classification_chain = pts["classify"] | llm | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

review = input("Type your review: ")
result = chain.invoke({"feedback": review})

# Output the result
print(result)
