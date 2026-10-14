# Arabic & Multilingual Aspect-Based Sentiment Analysis with MARBERT

A Natural Language Processing project for **aspect-based sentiment analysis (ABSA)** on Arabic and multilingual customer reviews.

The model detects which business aspects are mentioned in a review, then predicts the sentiment for each detected aspect.

Supported aspects include:

- Food
- Service
- Price
- Cleanliness
- Delivery
- Ambiance
- App experience
- General opinion

The project uses **MARBERT** because it is well suited for Arabic, dialectal Arabic, and informal social-media-style text.

---

## Project Overview

Traditional sentiment analysis usually predicts one overall label for a full review.

This project goes deeper by answering questions such as:

> The customer liked the food, but did they complain about the price?  
> Was the delivery experience negative even though the app experience was positive?

For each review, the model returns a structured JSON output like:

```json
{
  "review_id": "example_001",
  "aspects": ["food", "service", "price"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative",
    "price": "negative"
  }
}
```

---

## Key Features

- Arabic and dialectal Arabic support
- Mixed Arabic-English review handling
- Egyptian Arabic and informal review examples
- Aspect detection and sentiment classification
- JSON output for dashboards, APIs, or competition submissions
- Gradio demo interface for public presentation
- Kaggle-friendly training workflow

---

## Model Approach

The task is transformed into a pairwise classification problem.

For every review, the model checks each possible aspect separately.

Example:

```text
Review:
الخدمة ممتازة والأكل سيء

Aspect:
food
```

The model predicts one of the following labels:

- `not_mentioned`
- `positive`
- `negative`
- `neutral`

This allows the system to identify both:

1. Whether an aspect is mentioned
2. The sentiment toward that aspect

---

## Repository Structure

```text
arabic-multilingual-absa-marbert/
│
├── README.md
├── requirements.txt
├── multilingual-nlp-marbert.ipynb
├── gradio_absa_demo_app.py
├── demo_examples.md
└── .gitignore
```

Recommended optional folders:

```text
images/
outputs/
```

Do not upload large trained model files or full Kaggle datasets directly to GitHub.

---

## Dataset

The project was developed on Kaggle using the DeepX NLP dataset.

Expected Kaggle dataset path:

```python
DATA_DIR = "/kaggle/input/datasets/kareemmaged/deepx-nlp"
```

Expected files:

```text
train_fixed(Sheet1).csv
validation_fixed(Sheet1).csv
unlabeled_fixed(Sheet1).csv
DeepX_hidden_test (4).xlsx
```

The training and validation files contain labeled aspect and sentiment columns.

The unlabeled and hidden test files contain review text and metadata, and the trained model generates the predictions.

---

## How to Run on Kaggle

### 1. Open the notebook

Upload or open:

```text
multilingual-nlp-marbert.ipynb
```

### 2. Enable Kaggle settings

In the Kaggle notebook settings:

```text
Accelerator: GPU
Internet: On
```

Internet is needed for downloading the pretrained MARBERT model and for launching a public Gradio demo link.

### 3. Install dependencies

The notebook includes an installation cell, but you can also install dependencies manually:

```bash
pip install -r requirements.txt
```

### 4. Check dataset paths

Inside the notebook, make sure the paths match your Kaggle dataset location:

```python
DATA_DIR = "/kaggle/input/datasets/kareemmaged/deepx-nlp"

TRAIN_FILE = f"{DATA_DIR}/train_fixed(Sheet1).csv"
TEST_FILE = f"{DATA_DIR}/validation_fixed(Sheet1).csv"
```

For final prediction, change `TEST_FILE` to:

```python
TEST_FILE = f"{DATA_DIR}/unlabeled_fixed(Sheet1).csv"
```

or:

```python
TEST_FILE = f"{DATA_DIR}/DeepX_hidden_test (4).xlsx"
```

### 5. Run all notebook cells

The notebook will:

1. Load the dataset
2. Preprocess review text
3. Convert reviews into aspect-level training rows
4. Fine-tune MARBERT
5. Evaluate on validation data
6. Tune the aspect detection threshold
7. Save the best model and prediction artifacts

The trained model is saved to:

```text
/kaggle/working/marbert_absa_output/best_model
```

The metadata/artifacts file is saved to:

```text
/kaggle/working/marbert_absa_output/artifacts.json
```

---

## Gradio Demo

The repository includes a Gradio interface:

```text
gradio_absa_demo_app.py
```

The demo allows users to paste a review and see:

- Detected aspects
- Sentiment for each aspect
- Confidence scores
- Final JSON prediction
- All-aspect confidence table

To run it on Kaggle after training:

```bash
python gradio_absa_demo_app.py
```

The app should launch using:

```python
demo.queue().launch(share=True, debug=True)
```

`share=True` creates a public demo link that can be used for recording or sharing.

---

## Demo Examples

### Arabic Review

```text
الخدمة كانت محترمة جدا والمكان رايق، بس الأكل وصل بارد واللحمة ناشفة والسعر مبالغ فيه
```

Expected behavior:

- Service: positive
- Ambiance: positive
- Food: negative
- Price: negative

### Mixed Arabic and English

```text
The app is smooth and checkout was fast, but el delivery اتأخر ساعتين والدعم مردش عليا خالص.
```

Expected behavior:

- App experience: positive or neutral
- Delivery: negative
- Service: negative

### Egyptian Arabic Franko

```text
el akl kan gamed awy bas el service slow w el bill kan aghla mn el expected.
```

Expected behavior:

- Food: positive
- Service: negative
- Price: negative

---

## Output Format

The model returns predictions in this format:

```json
{
  "review_id": "review_001",
  "aspects": ["food", "service"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative"
  }
}
```

If no aspect is detected, the output becomes:

```json
{
  "review_id": "review_001",
  "aspects": ["none"],
  "aspect_sentiments": {}
}
```

---

## Notes and Limitations

- The model is strongest on Arabic-script and dialectal Arabic reviews.
- Franco-Arabic examples may work, but performance depends on how much similar text exists in the training data.
- Very short or sarcastic reviews can be harder to classify.
- The detection threshold affects how many aspects are included in the final output.
- The trained model files are not included in this repository because they can be large.

---

## Author

Created by **Mohammed Ibrahim** as part of a multilingual NLP project focused on Arabic customer review understanding.
