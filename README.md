# multimodal-captioning-via-clip-gpt2

This repository implements a lightweight multimodal image captioning system that integrates CLIP as a visual encoder and GPT-2 as a language decoder. By connecting these models via a linear projection module, we enable prompt-conditioned caption generation without fine-tuning either backbone.

This project was developed as part of the Peking University Summer Lab 2025 by Xiangyi Liu (Liaoning University).

## Overview

The system generates natural language captions based on image input through the following components:

- **CLIP (ViT-B/16)**: used to extract visual embeddings from input images.
- **Linear Projection Layer**: transforms image embeddings to match GPT-2’s input space.
- **GPT-2**: serves as an autoregressive decoder to generate captions from embeddings and prompts.

The captions are evaluated using CLIP similarity scores and further analyzed using entropy-weighted metrics for prompt effectiveness.

## Features

- Supports image-conditioned caption generation without modifying pretrained GPT-2.
- Enables evaluation of prompt quality through CLIP similarity and entropy-weighted scoring.
- Includes visualizations such as CLIP score distributions, token-length impact, and prompt comparison.
- Designed for modular experimentation and integration with additional models.

## Model Architecture
Input Image → CLIP Encoder → Linear Projection → [Image Embedding + Prompt] → GPT-2 → Caption Output

- CLIP outputs a 512-dimensional image embedding.
- The linear layer projects the embedding into GPT-2’s token embedding space.
- The projected embedding is prepended to the tokenized prompt.
- GPT-2 performs greedy decoding to generate up to 30 tokens.

## Environment

- Python 3.10
- PyTorch 2.1.2
- Hugging Face Transformers 4.42.0
- openai-clip
- Hardware: NVIDIA A10 GPU (24GB), 16-core CPU, 60GB RAM
- OS: Ubuntu 20.04


## Repository Structure

```text
Emu3-Mutimodal-NextToken/
├── data/
│   ├── images/
│   ├── processed/
│   └── raw/
│       └── captions_train2017.json
├── gpt2_epoch8/
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── notebook/
│   └── results_done.ipynb
├── src/
│   ├── align_tokens.py
│   ├── app.py
│   ├── build_input.py
│   ├── dataset.py
│   ├── inference.py
│   ├── predictor.py
│   ├── prepare_data.py
│   ├── test.py
│   ├── testfinal.py
│   ├── train.py
│   └── utils.py
├── templates/
│   └── index.html
└── test_images/
    └── aligned_tokens_full.jsonl
```
---

### Directory & File Descriptions

- **`data/`**  
  Root directory for all dataset files.
  - `images/`: Raw COCO images used for caption generation.
  - `processed/`: Stores token-aligned multimodal input data in `.jsonl` format.
  - `raw/`: Contains the original COCO 2017 annotation file `captions_train2017.json`.

- **`gpt2_epoch8/`**  
  Stores a fine-tuned GPT-2 model checkpoint and tokenizer configuration.
  - `model.safetensors`: Weights of the GPT-2 model after 8 epochs of training.
  - `config.json`, `tokenizer_config.json`: Architecture and tokenizer settings.
  - `vocab.json`, `merges.txt`: Files for Byte Pair Encoding (BPE) tokenizer.
  - `generation_config.json`: Controls decoding parameters like temperature and top-k.
  - `special_tokens_map.json`: Maps tokens like `<pad>` or `<eos>` for generation.

- **`notebook/`**  
  Contains post-processing analysis.
  - `results_done.ipynb`: Jupyter notebook for evaluating caption quality, plotting metrics.

- **`src/`**  
  All Python scripts for data processing, training, and inference.
  - `prepare_data.py`: Extracts image-caption pairs from raw COCO files.
  - `align_tokens.py`: Encodes CLIP image embeddings and aligns them with tokenized text.
  - `build_input.py`: Constructs `[image_tokens] + text_tokens` format for training.
  - `dataset.py`: Defines the PyTorch `Dataset` for multimodal inputs.
  - `train.py`: Fine-tunes GPT-2 on the aligned multimodal dataset.
  - `inference.py`: Generates captions using the trained GPT-2 model.
  - `predictor.py`: Utility for batch predictions or confidence scoring.
  - `utils.py`: Contains helper functions for tokenization, file I/O, etc.
  - `test.py`, `testfinal.py`: Scripts for evaluation or testing, possibly legacy or exploratory.
  - `app.py`: (Optional) Flask or Gradio-based demo interface for interactive use.

- **`templates/`**  
  Frontend component for the optional web-based demo.
  - `index.html`: HTML layout for image upload and caption output.

- **`test_images/`**  
  Includes aligned input for evaluation.
  - `aligned_tokens_full.jsonl`: Tokenized test samples in the final input format.

---

## Recommended Execution Order

To ensure reproducibility and a clear workflow, we recommend following the steps below in the specified order. This pipeline assumes the availability of raw COCO-style image-caption data and guides users from data preprocessing to model training and evaluation.

1. **`prepare_data.py`**  
   Extracts and organizes image-caption pairs from the COCO annotation JSON.

2. **`align_tokens.py`**  
   Encodes CLIP embeddings and aligns them with tokenized captions.

3. **`build_input.py`**  
   Constructs the `[image_tokens] + text_tokens` training format.

4. **`train.py`**  
   Fine-tunes GPT-2 on the constructed multimodal dataset.

5. **`inference.py`**  
   Generates captions using the fine-tuned GPT-2 model.

6. **`results_done.ipynb`**  
   Visualizes results (e.g., CLIP scores, token distributions, prompt quality).

7. *(Optional)* **`app.py`**  
   Launches a simple demo for interactive captioning via web UI.

---

## License

This repository is licensed under the MIT License.

---

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) for visual-semantic embedding
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for GPT-2 and model utilities
- COCO Dataset (2017) for image-caption supervision



## Visualizations

- Histogram of CLIP score distribution
- Scatter plot of token count vs. CLIP score
- Cumulative CLIP score comparison (Top-50 vs Bottom-50)
- Prompt-prompt compatibility matrix
- Normalized entropy-weighted prompt rankings

Plots and raw data are available in the `results/` directory and are embedded in the accompanying LaTeX report.

## Limitations

- Only supports CLIP (ViT-B/16) and GPT-2 (small)
- Prompts are manually selected and not optimized
- Evaluation does not include human-annotated metrics such as BLEU or CIDEr

## Future Work

- Integrate larger or instruction-tuned models (e.g., BLIP-2, LLaVA)
- Add automatic prompt optimization mechanisms
- Incorporate human feedback or perception-aligned metrics
- Extend the system to support multilingual captionin



