# Normalize Model Name Access for LLMs
def get_model_name(llm) -> str:
    """
    Return the model name from an LLM object (Gemini, DeepSeek, etc.),
    falling back safely if attribute isn't found.
    """
    for attr in ["model_name", "model"]:
        if hasattr(llm, attr):
            return getattr(llm, attr)
    return "UnknownModel"