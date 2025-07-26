from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Create a ChatGooleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 1. Base prompt: list features
base_p = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Hubble, an expert astrophysicist. Answer physics and astrophysics questions clearly and accurately, adjusted for any skill level.",
        ),
        ("human", "Hey Hubble, can you help me with——{q}?"),
    ]
)


# 2. Helper to build a pros/cons (or any) sub‑chain
# `human_prompt` must include a "{features}" placeholder
def make_branch(human_prompt: str):
    pt = ChatPromptTemplate.from_messages([
        ("system", "You are Hubble, an expert astrophysicist. Answer questions clearly at any level."),
        ("human", human_prompt),
    ])

    return (
        # Same effect: RunnableLambda(lambda feat: pt.format_prompt(features=feat)) # type: ignore
        RunnableLambda(lambda feat: pt.invoke({"features":feat})) # type: ignore
        | llm
        | StrOutputParser()
    )

# 3. Define two branches: pros & cons
pros_chain = make_branch(human_prompt="Given these features: {features}, list their pros.")
cons_chain = make_branch(human_prompt="Given these features: {features}, list their cons.")

# 4. Run both in parallel and merge results
parallel = RunnableParallel(branches={"pros": pros_chain, "cons": cons_chain})
merge = RunnableLambda(lambda out: f"Pros:\n{out['branches']['pros']}\n\nCons:\n{out['branches']['cons']}")  # type: ignore

# 5. Full pipeline
chain = (
    base_p
    | llm
    | StrOutputParser()  # produces feature list
    | parallel  # runs pros & cons in parallel
    | merge  # formats final text
)

if __name__ == "__main__":
    print(chain.invoke({"q": "Quasi-Zenith Satellite System (QZSS)"}))
