"""
Model Mappings

This file provides mappings between short model names and their full Hugging Face model IDs
for text embedding models used in this project.
"""

MODEL_MAPPINGS = {
    # Sentence Transformers models
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "sentence-t5-base": "sentence-transformers/sentence-t5-base",
    
    # Specialized embedding models
    "ConTeXT-Skill-Extraction-base": "TechWolf/ConTeXT-Skill-Extraction-base",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "granite-embedding-30m-english": "ibm-granite/granite-embedding-30m-english",
    "stella_en_400M_v5": "NovaSearch/stella_en_400M_v5",
    "stella_en_1.5B_v5": "NovaSearch/stella_en_1.5B_v5",
    "jasper_en_vision_language_v1": "NovaSearch/jasper_en_vision_language_v1",
    "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "e5-large-v2": "intfloat/e5-large-v2",
    "bge-m3": "BAAI/bge-m3",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "instructor-large": "hkunlp/instructor-large",
    "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "Jina-embeddings-v3": "jinaai/jina-embeddings-v3",
}

def get_model_id(model_name):
    """
    Get the full Hugging Face model ID for a given short model name.
    
    Args:
        model_name (str): Short model name
        
    Returns:
        str: Full Hugging Face model ID
    """
    if model_name in MODEL_MAPPINGS:
        return MODEL_MAPPINGS[model_name]
    elif model_name in MODEL_MAPPINGS.values():
        # If the full model ID was passed in, just return it
        return model_name
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_MAPPINGS.keys())}")

def list_available_models():
    """
    List all available model names and their full Hugging Face model IDs.
    
    Returns:
        list: List of dictionaries with 'name' and 'id' keys
    """
    return [{"name": name, "id": model_id} for name, model_id in MODEL_MAPPINGS.items()]

def print_available_models():
    """Print a formatted list of all available models."""
    print("Available embedding models:")
    print("-" * 80)
    print(f"{'Short Name':<20} | {'Full Model ID':<50}")
    print("-" * 80)
    
    for name, model_id in sorted(MODEL_MAPPINGS.items()):
        print(f"{name:<20} | {model_id:<50}")

if __name__ == "__main__":
    print_available_models() 