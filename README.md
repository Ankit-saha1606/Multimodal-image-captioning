# Multimodal Image Captioning with BLIP

A vision-language project that fine-tunes **BLIP** (Bootstrapping Language-Image Pre-training) to generate natural-language captions for images, using the **Flickr8k** dataset. The goal was to build and understand a complete captioning pipeline — data preparation, efficient fine-tuning, quantitative evaluation, and an interactive demo — rather than just running a pretrained model out of the box.

## Why BLIP + LoRA

BLIP combines a vision encoder with a text decoder and is pretrained on large-scale image-text pairs, which makes it a strong starting point for captioning. Fully fine-tuning it is expensive, so this project uses **LoRA (Low-Rank Adaptation)** to update only a small set of additional parameters, keeping training fast and memory-efficient while still adapting the model to Flickr8k's style of captions.

## Project structure

```
Multimodal-image-captioning/
├── notebooks/
│   └── image_captioning_pipeline.ipynb   # end-to-end training & eval notebook
├── results/
│   ├── loss_curve.png                    # training vs. validation loss
│   └── sample_captions/                  # example model outputs on unseen images
├── requirements.txt
└── README.md
```
*(Update the paths above if your actual repo layout differs.)*

## Pipeline

1. **Setup** — install dependencies, load a GPU runtime.
2. **Data preparation** — parse Flickr8k captions, resize/normalize images to 224×224.
3. **Model** — load `Salesforce/blip-image-captioning-base` and attach LoRA adapters to the relevant layers.
4. **Training** — fine-tune with Hugging Face's `Seq2SeqTrainer`, using early stopping on validation loss.
5. **Decoding** — generate captions at inference time using beam search.
6. **Evaluation** — score generated captions against references using standard captioning metrics.

## Results




![Training vs validation loss](results/loss_curve.png)

## Sample outputs

A few example generated captions on held-out images are in [`results/sample_captions/`](results/sample_captions).

## Getting started

```bash
git clone https://github.com/Ankit-saha1606/Multimodal-image-captioning.git
cd Multimodal-image-captioning
pip install -r requirements.txt
```

Then open the notebook in `notebooks/` to reproduce training, or run the Gradio demo (if included) for interactive inference.

## Requirements

- torch
- transformers
- accelerate
- peft (for LoRA)
- pandas, matplotlib
- nltk, rouge-score, evaluate
- gradio (for the demo)

See `requirements.txt` for exact versions.

## Possible extensions

- Fine-tune on a different, less commonly used dataset to differentiate from typical Flickr8k demos.
- Add a visual question answering (VQA) head alongside captioning.
- Compare BLIP against BLIP-2 or a ViT-GPT2 baseline on the same data.

## Acknowledgements

- [BLIP (Salesforce)](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## License

MIT License — free to use and adapt for research or educational purposes.
