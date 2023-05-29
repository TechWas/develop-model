# Model Development

This repository focuses on developing models for the machine learning part of TechWas. We experiment with transfer learning, transformers, and other techniques to find the best output for our tasks.

## Tasks

Our project focuses on two main tasks:

- Image classification: Our goal is to accurately classify images into 15 predefined categories.
- Text summarization: Our goal is to generate concise and informative summaries of longer text of article.

## Approach

To achieve our goals, we use transfer learning techniques to leverage pre-trained models and fine-tune them with our own dataset for our specific tasks. This allows us to quickly develop high-performing models without the need for extensive training data or computational resources.

## Final Model

- [image classifiation](task_comp-vis/e-waste_xception_v1.ipynb)
- [text summarization](task_nlp/finetuning_bartlargecnn.ipynb)

## Preview

You can preview the implementation in this [web app](https://techwas-ml-preview.streamlit.app/)!

## Todos

- [ ] text classification model (optional)
- [ ] object detection model (optional)

## Model Architecture

- Image Classification

  - Base model
    ![base](docs/img/img-base.png)

  - Fine tuned
    ![fine-tuned](docs/img/img-fine-tuned.png)

- Text Summarization

  ![transformers](docs/img/bart.png)
