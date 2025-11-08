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

This project supports two Hugging Face sentence-transformer models for embedding and storing document vectors.

### `all-MiniLM-L6-v2`
- Free and lightweight — ideal for experimentation and learning  
- Provides a strong balance between speed and accuracy  
- Recommended for small- to medium-scale use  

### `sentence-transformers/all-mpnet-base-v2`
- More advanced model with higher semantic precision and contextual understanding  
- Slightly slower but better suited for production or complex retrieval tasks  

### Why the switch?
The project originally used **MiniLM-L6-v2** for its speed and simplicity.  
It later adopted **MPNet-base-v2** to improve retrieval accuracy and handle more complex semantic queries.

### References
See examples in:
- `4_rag/1a_build_vectorstore.py` – Load and embed documents  
- `4_rag/1b_query_vectorstore.py` – Query stored vectors  

Further reading:
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

