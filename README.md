# langchain-playbook

A growing collection of LangChain patterns and implementations inspired by @bhancockio’s crash course, with documented learnings and extensions.

## Model Information

This project primarily uses the **Google Gemini API** via the `langchain-google-genai` integration.

- Obtain your API key from: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Use the model: `gemini-2.0-flash`
- Best suited for: text generation, chat-based applications, and multimodal tasks

As an alternative, the project also supports **DeepSeek Chat** via **OpenRouter**, which is compatible with LangChain’s `ChatOpenAI` interface.

- Obtain your API key from: [https://openrouter.ai](https://openrouter.ai)
- Use the model: `deepseek/deepseek-chat-v3-0324:free`
- Supports OpenAI-compatible interfaces

Both models are freely accessible, making them ideal for experimentation and learning without usage fees.

> You may also substitute other LLM providers (OpenAI, Anthropic, Mistral, etc.) by configuring your API keys and model names accordingly.

For usage examples, refer to the scripts under `1_chat_models/` in this repository:
- `1_basic.py` for a simple Gemini-based prompt
- `3_alternatives.py` to compare Gemini and DeepSeek via OpenRouter

## Vector Embeddings via Hugging Face

For embedding and storing document vectors, the project uses the Hugging Face model: `all-MiniLM-L6-v2`.

- Free to use locally
- Provides a good trade-off between performance and accuracy
- Suitable for experimentation and learning

To see how this embedding model is used with ChromaDB and LangChain, refer to:
- `scripts/1a_build_vectorstore.py` – Load and embed documents
- `scripts/1b_query_vectorstore.py` – Query stored vectors


For more information on the Hugging Face embedding model, visit:  
[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
