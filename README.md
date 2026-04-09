# EWE-1 Inference Repository

This repository contains the inference pipeline for the [**EWE-1** family of models](https://ewe-1.com/), created by [sistemalabs](https://sistemalabs.com/). EWE-1-slim is a suite of causal transformer models built with [**sequifier**](https://github.com/0xideas/sequifier) that project Ethereum transaction histories into a rich, forward-looking embedding space.

By analyzing a look-back window of 64 transactions across 31 contextual, behavioral, and temporal features, these models generate high-dimensional vectors optimized to predict future wallet behavior. These embeddings are ideal for downstream tasks like fraud detection, credit scoring, and user segmentation.

---

## Getting Started 🐥

### Prerequisites
1. Ensure you have Python installed with the necessary dependencies: `pandas`, `numpy`, and `sequifier`.
2. Place the pre-trained EWE-1 ONNX model file into a `models/` directory at the project root (e.g., `models/ewe1-slim-small.onnx`). [*Models are available via huggingface.*](https://huggingface.co/sistemalabs)
3. Place your raw transaction history files in a `data/` directory. Each file must be a `.parquet` file named after the target wallet address (e.g., `data/0x123...abc.parquet`).

### Running Inference
To run the full pipeline and generate embeddings, simply execute the main script from the root of the repository (without the onnx file extension):

```bash
python generate_embedding.py [MODEL_NAME]
```

---

## Model Variants 🐑

The EWE-1 family includes three model sizes. The repo is configured by default for the **Small** variant.

| Model | Embedding Dimension | Attention Layers |
| :--- | :--- | :--- |
| **EWE-1-slim-small** | 384 | 12 |
| **EWE-1-slim-medium** | 768 | 16 |
| **EWE-1-slim-large** | 1536 | 24 |

*Note: All models share a window size of 64 and 16 attention heads. They were trained on 1.1 billion transaction records from 2024 and 2025.*

### Pipeline Internals
When you run the script, the following sequence occurs automatically:
1. **Feature Extraction:** Reads raw transaction parquets from `data/` and applies the 31-feature extraction logic.
2. **Aggregation:** Concatenates all processed wallets into a temporary file (`temp-data/feature-data.parquet`).
3. **Preprocessing:** Executes `sequifier preprocess` to format the data into the 64-transaction sliding window format.
4. **Inference:** Executes `sequifier infer` to pass the preprocessed data through the transformer model.
5. **Output Generation:** Splits the resulting embeddings back out by sequence ID and saves them to the `embeddings/` directory.
