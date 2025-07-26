# Docs: https://python.langchain.com/docs/concepts/prompt_templates/
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


def print_section(title, content):
    print(f"\n** {title} **\n{content}")


# Section I: Single Placeholder Template
template = "Tell me why {topic}"

# Creates a ChatPromptTemplate from a plain template string with placeholders.
prompt_single = ChatPromptTemplate.from_template(template)
result = prompt_single.invoke({"topic": "airplanes fly"})
print_section("Prompt from Template", result)


# Section II: Multiple Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multi = ChatPromptTemplate.from_template(template_multiple)
result = prompt_multi.invoke({"adjective": "funny", "animal": "pandas"})
print_section("Prompt with Multiple Placeholders", result)


# Section III: Prompt with system and human messages using tuples
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

# from_messages() builds a structured chat prompt using role-specific message templates.
prompt_msg = ChatPromptTemplate.from_messages(messages)
result = prompt_msg.invoke({"topic": "pandas", "joke_count": 3})
print_section("Prompt with System and Human Messages (Tuple)", result)


# Section IV: Mixing tuples and static HumanMessage objects
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    # This HumanMessage is treated as static — no template parsing is done here.
    HumanMessage(content="Tell me  jokes."),
]
prompt_mixed = ChatPromptTemplate.from_messages(messages)
result = prompt_mixed.invoke({"topic": "pandas"})
print_section("Prompt with Tuple and Core Messages", result)

# NOTE:
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes.")
# Here, placeholders like {joke_count} inside HumanMessage won't be replaced.
# For dynamic values, use the ("role", "template string") format instead.
