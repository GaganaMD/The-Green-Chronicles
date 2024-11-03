from transformers import AutoTokenizer, AutoModel, pipeline

def load_tokenizer(model_name="bert-base-uncased"):
    """
    Load a tokenizer from a given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model(model_name="bert-base-uncased"):
    """
    Load a model from a given model name.
    """
    model = AutoModel.from_pretrained(model_name)
    return model

def load_text_generation_pipeline(model_name="distilgpt2"):
    """
    Load a text generation pipeline with a given model.
    """
    pipe = pipeline("text-generation", model=model_name)
    return pipe

# Example usage:
# tokenizer = load_tokenizer("bert-base-uncased")
# model = load_model("bert-base-uncased")
# generator = load_text_generation_pipeline("openai-community/gpt2")
