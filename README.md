# Large Language Model-based Dialogue Embeddings for User Modeling

This is the repository for the thesis "Large Language Model based Dialogue Embeddings for User Modeling".

Python reference implementations for synthetic dataset generation, fine-tuning,
retrieval evaluation, zero-shot dataset classification and plotting code are provided.

## Scripts

### openai_synthesize.py
Generates synthetic data (e.g., instructions or dialogs) from personas using prompt templates and a language model API. Outputs JSONL.

### train_sbert_v3.py
Fine-tunes a Transformer Hugging Face formatted LLM with a contrastive objective; supports optional evaluation dataset and saves checkpoints/ final model.

### input_instruction_retrieval.py
Evaluates how well instruction embeddings retrieve their corresponding input/persona embeddings. Reports Top-1/Top-5 accuracy and MRR.

### data_classifier.py
Performs zero-shot classification of `instruction` and `input` fields into high-level categories and writes results back to the dataset.

### tsne_plot_embeddings.py, umap_plot_embeddings.py
Produces 2D visualizations of instruction and persona embeddings using t-SNE or UMAP.

### model_mappings.py
Maps short model names to full Hugging Face model IDs.

### prompt_templates.py

Stores prompt string templates used during synthesis and dialog generation.

## Note

These scripts are provided only as reference. They are not intended to be run as is.

