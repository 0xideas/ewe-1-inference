# EWE-1 Inference Repository

This repository contains the inference pipeline for the [**EWE-1** family of models](https://ewe-1.com/), created by [sistemalabs](https://sistemalabs.com/). EWE-1-slim is a suite of causal transformer models built with [**sequifier**](https://github.com/0xideas/sequifier) that project Ethereum transaction histories into a rich, forward-looking embedding space.

By analyzing a look-back window of 64 transactions across 31 contextual, behavioral, and temporal features, these models generate high-dimensional vectors optimized to predict future wallet behavior. These embeddings are ideal for downstream tasks like fraud detection, credit scoring, and user segmentation.

**This utility deletes intermediate folders, make sure not to mix it with pre-existing folder structures where data might inadvertedly be deleted. It is designed as a standalone utility that should live in its own folder**

---

## Getting Started 🐥

### Prerequisites
1. Clone this repository
2. Ensure you have Python installed with `sequifier>=1.1.1.3` installed (run `pip install sequifier` if not), and that you are on Mac or Linux.
3. Place the pre-trained EWE-1 ONNX model file into a `models/` directory at the project root (e.g., `models/ewe1-slim-small.onnx`). [*Models are available via huggingface.*](https://huggingface.co/sistemalabs)
4. Place your raw transaction history files in a `data/` directory. Each file must be a `.parquet` file named after the target wallet address (e.g., `data/0x123...abc.parquet`).

### Running Inference

To run the full pipeline and generate embeddings, simply execute the main script from the root of the repository (without the onnx file extension):

```bash
python generate_embedding.py \<MODEL_NAME\> [N_LAST]
```

The second, optional argument `N_LAST` specifies the number sequence values that you want to generate embeddings for. If this argument is not provided, embeddings will be calculated for all sequence values from the 64th position onwards, and if it is, embeddings will be calculated for the `N_LAST` last sequence positions.

---

## Model Variants 🐑

The EWE-1 family includes three model sizes.

| Model | Embedding Dimension | Attention Layers | Link |
| :--- | :--- | :--- | :--- |
| **EWE-1-slim-small** | 384 | 12 | [huggingface model card](https://huggingface.co/sistemalabs/EWE-1-slim-small) |
| **EWE-1-slim-medium** | 768 | 16 | [huggingface model card](https://huggingface.co/sistemalabs/EWE-1-slim-medium) |
| **EWE-1-slim-large** | 1536 | 24 | [huggingface model card](https://huggingface.co/sistemalabs/EWE-1-slim-large) |

*Note: All models share a window size of 64 and 16 attention heads. They were trained on 1.1 billion transaction records from 2024 and 2025.*

## Input Data 💽

The full transaction history for each user that you want to create embeddings for should be stored in chronological ascending order in a parquet file names `[USER_ADDRESS].parquet` and put in the data folder. Each user should be in a separate parquet file.

The transaction history of a given user should look like this:

| index | block_timestamp | transaction_index | from_address | to_address | nonce | input | gas | receipt_gas_used | gas_price | receipt_status | receipt_contract_address | transaction_type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 2025-12-30 21:56:11 | 141 | 0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1 | 0xa5fa7675ce8c740c022fb3aab248dfd7a097d3ad | 630125 | 0x | 23520 | 21000 | 45373308 | 1 | None | 2 |
| **1** | 2025-12-30 21:56:11 | 142 | 0x5babe600b9fcd5fb7b66c0611bf4896d967b23a1 | 0xe0803fc64311e530dc1bac8dd1b20cef881f6cc5 | 630126 | 0x | 23520 | 21000 | 45373308 | 1 | None | 2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Pipeline Internals 🔎

When you run the script, the following sequence occurs automatically:
1. **Feature Extraction:** Reads raw transaction parquets from `data/` and applies the 31-feature extraction logic.
2. **Aggregation:** Concatenates all processed wallets into a temporary file (`temp-data/eth-sequences-filtered.parquet`).
3. **Preprocessing:** Executes `sequifier preprocess` to format the data into the 64-transaction sliding window format.
4. **Inference:** Executes `sequifier infer` to pass the preprocessed data through the transformer model.
5. **Output Generation:** Splits the resulting embeddings back out by sequence ID and saves them to the `embeddings/` directory.
