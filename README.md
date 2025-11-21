# Langchain Playbook

A growing collection of **LangChain patterns and implementations**, inspired by @bhancockio’s crash course, with documented learnings and extensions. This repository is designed for experimentation with different LLMs, embeddings, and web/document integrations.

---

## Model Information

This project primarily uses the **Google Gemini API** via the `langchain-google-genai` integration.

* Obtain your API key from: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
* Use the model: `gemini-2.0-flash`
* Best suited for: text generation, chat-based applications, and multimodal tasks

As an alternative, the project also supports **DeepSeek Chat** via **OpenRouter**, compatible with LangChain’s `ChatOpenAI` interface.

* Obtain your API key from: [https://openrouter.ai](https://openrouter.ai)
* Use the model: `deepseek/deepseek-chat-v3-0324:free`
* Supports OpenAI-compatible interfaces

> You may also substitute other LLM providers (OpenAI, Anthropic, Mistral, etc.) by configuring your API keys and model names accordingly.

### Usage Examples

Refer to the scripts under `1_chat_models/`:

* `1_basic.py` – simple Gemini-based prompt
* `2_basic_conversation.py` – basic conversational example
* `3_alternatives.py` – compare Gemini and DeepSeek via OpenRouter
* `4_realtime_chat.py` – real-time streaming chat example
* `5_firestore_realtime_chat.py` – Firestore-based real-time chat

---

## Prompt Templates

Scripts under `2_prompt_templates/` show reusable prompt patterns:

* `1_basic.py` – simple template example
* `2_realtime_chat.py` – prompts for real-time streaming chat

---

## Chains

Examples under `3_chains/` demonstrate different chain patterns:

* `1_basic.py` – single-step chain
* `2_runnables.py` – using Runnables
* `3_extending_chain.py` – extending chains with custom logic
* `4_runnable_parallel_chains.py` – parallel chain execution
* `5_runnable_branch.py` – branching chain execution

---

## RAG (Retrieval-Augmented Generation)

Scripts in `4_rag/` show embedding, vector store, and retrieval examples using Hugging Face embeddings.

---

## Agents and Tools

### Agents (`5_agent_and_tools/agents`)

* `1_react_chat.py` – basic React-style agent
* `2_react_chat_docstore.py` – agent connected to a document store

### Tools (`5_agent_and_tools/tools`)

* `1_tool_constructor.py` – build tools dynamically
* `2_tool_decorator.py` – tool creation with decorators
* `3_base_tool.py` – base class for tools

`1_basic.py` – basic agent and tool example

---

## Vector Embeddings via Hugging Face

This project supports two Hugging Face sentence-transformer models:

### `all-MiniLM-L6-v2`

* Lightweight and free — ideal for experimentation
* Balanced speed and accuracy
* Recommended for small- to medium-scale use

### `sentence-transformers/all-mpnet-base-v2`

* Higher semantic precision and contextual understanding
* Slightly slower but better for production or complex retrieval tasks

> The switch from MiniLM-L6-v2 to MPNet-base-v2 improves retrieval accuracy for complex semantic queries.

Examples:

* `4_rag/1a_build_vectorstore.py` – Load and embed documents
* `4_rag/1b_query_vectorstore.py` – Query stored vectors

Further reading:

* [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

---

## Additional Tools

### Firebase Firestore
This project supports persistent chat history using Firebase Firestore through LangChain's FirestoreChatMessageHistory. This allows you to store and retrieve chat messages across sessions, enabling real-time or historical conversation tracking.

Features:
* Store each message in Firestore with metadata
* Retrieve previous conversation messages for context in multi-turn chats
* Integrates seamlessly with ChatOpenAI or Google Gemini API calls

### Tavily

**[Tavily](https://docs.tavily.com/documentation/quickstart)** is a lightweight workflow and orchestration tool designed for automation of repetitive tasks in LangChain projects.

* Allows for clean workflow definitions in YAML/JSON
* Supports scheduling, logging, and conditional execution
* Integrates seamlessly with LangChain agents and LLM calls

Example usage:

```python
from tavily import TavilyClient
client = TavilyClient("tvly-dev-********************************")
response = client.search(
    query=""
)
print(response)
```

### Firecrawl

**[Firecrawl](https://docs.firecrawl.dev/introduction)** is a document and web crawler designed to collect content for RAG (Retrieval-Augmented Generation) pipelines.

* Can crawl websites, PDFs, or local directories
* Automatically cleans and formats content for embedding
* Works with Hugging Face embeddings for vector storage

Example usage:

```python
from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key="fc-YOUR-API-KEY")

# Scrape a website:
doc = firecrawl.scrape("https://firecrawl.dev", formats=["markdown", "html"])
print(doc)
```

> Tavily and Firecrawl combined allow building fully automated RAG pipelines with minimal boilerplate.

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/maheshus/langchain-playbook.git
cd langchain-playbook
```

## 2. Install dependencies using Poetry

```bash
poetry install
```

## 3. Activate the Poetry virtual environment

```bash
poetry env activate
```

> This will print the command to activate the environment (for example, a batch file on Windows). Run that command in your terminal to activate the environment.

## 4. Add your API keys

Create a `.env` file in the project root with the following:

```bash
GOOGLE_GENAI_API_KEY="your-key-here"
OPENROUTER_API_KEY="your-key-here"
TAVILY_API_KEY="your-key-here"
FIRECRAWL_API_KEY="your-key-here"
```

## 5. Run example scripts

Inside the activated environment, run:

```bash
python 1_chat_models/1_basic.py
```
---

## Contribution

Contributions are welcome! Feel free to open issues, submit PRs, or add new patterns and tools for LLM experimentation.

---

## License

MIT

---

## Author

Created and maintained by Mahesh U S
