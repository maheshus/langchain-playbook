# langchain-playbook

A growing collection of LangChain patterns and implementations from @bhancockio's crash course, with documented learnings and extensions.

## Model Info

This project primarily uses the **Google Gemini API** via the `langchain-google-genai` integration.

- 🔑 Get your API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- 🧩 Use the model `gemini-2.0-flash`
- ✅ Works well for text generation and multimodal tasks

---

As an alternative, this project also supports the **DeepSeek Chat model** via **OpenRouter** (OpenAI-compatible).

- 🔑 Get your API key from [openrouter.ai](https://openrouter.ai)
- 🧩 Use the model `deepseek/deepseek-chat-v3-0324:free`
- ✅ Fully compatible with LangChain’s `ChatOpenAI` class

---

💡 Both models are **free to use**, making them ideal for experimenting and learning without worrying about usage limits or payments.

You’re welcome to use any other model (OpenAI, Claude, Mistral, LLaMA, etc.) — just update your API key and model name accordingly.
