"""
Gradio demo for Arabic / multilingual Aspect-Based Sentiment Analysis (ABSA).

How to use:
1) Train the notebook first so this folder exists: ./marbert_absa_output/best_model
2) Make sure artifacts.json exists in ./marbert_absa_output/ (your notebook saves it in section 35).
3) Install Gradio if needed: pip install -q gradio
4) Run: python gradio_absa_demo_app.py
"""

import os
import re
import ast
import json
import html
import unicodedata
from typing import Dict, Any, List

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import emoji
except Exception:  # keeps the app usable if emoji is not installed
    emoji = None


# -----------------------------
# Paths
# -----------------------------
OUTPUT_DIR = "./marbert_absa_output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
ARTIFACTS_PATH = os.path.join(OUTPUT_DIR, "artifacts.json")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Could not find {MODEL_DIR}. Train the notebook first, then run this app."
    )


# -----------------------------
# Load saved training artifacts
# -----------------------------
def load_artifacts() -> Dict[str, Any]:
    defaults = {
        "max_len": 96,
        "best_threshold": 0.30,
        "label2id": {
            "not_mentioned": 0,
            "positive": 1,
            "negative": 2,
            "neutral": 3,
        },
        "id2label": {
            0: "not_mentioned",
            1: "positive",
            2: "negative",
            3: "neutral",
        },
        "aspects": [
            "food",
            "service",
            "price",
            "cleanliness",
            "delivery",
            "ambiance",
            "app_experience",
            "general",
        ],
        "aspect_prompts": {
            "food": "الجانب: الطعام والأكل",
            "service": "الجانب: الخدمة والتعامل",
            "price": "الجانب: السعر والأسعار",
            "cleanliness": "الجانب: النظافة",
            "delivery": "الجانب: التوصيل",
            "ambiance": "الجانب: الأجواء والمكان",
            "app_experience": "الجانب: التطبيق وتجربة الاستخدام",
            "general": "الجانب: التقييم العام",
        },
    }

    if not os.path.exists(ARTIFACTS_PATH):
        return defaults

    with open(ARTIFACTS_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)

    defaults.update(saved)

    # JSON saves dictionary keys as strings; convert id2label keys back to int.
    defaults["id2label"] = {int(k): v for k, v in defaults["id2label"].items()}
    return defaults


artifacts = load_artifacts()
MAX_LEN = int(artifacts["max_len"])
DEFAULT_THRESHOLD = float(artifacts.get("best_threshold", 0.30))
ASPECTS = artifacts["aspects"]
ASPECT_PROMPTS = artifacts["aspect_prompts"]
LABEL2ID = artifacts["label2id"]
ID2LABEL = artifacts["id2label"]
OUTPUT_NONE_LABEL = "none"

ASPECT_DISPLAY = {
    "food": "Food / الطعام",
    "service": "Service / الخدمة",
    "price": "Price / السعر",
    "cleanliness": "Cleanliness / النظافة",
    "delivery": "Delivery / التوصيل",
    "ambiance": "Ambiance / الأجواء",
    "app_experience": "App Experience / تجربة التطبيق",
    "general": "General / عام",
}

SENTIMENT_DISPLAY = {
    "positive": "Positive 👍",
    "negative": "Negative 👎",
    "neutral": "Neutral 😐",
    "not_mentioned": "Not mentioned",
}

SENTIMENT_CLASS = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "not_mentioned": "muted",
}


# -----------------------------
# Same light preprocessing used in training
# -----------------------------
class ArabicPreprocessor:
    def __init__(
        self,
        remove_diacritics: bool = True,
        replace_urls: bool = True,
        replace_emails: bool = True,
        replace_mentions: bool = True,
        replace_hashtags: bool = False,
        keep_emojis: bool = True,
        reduce_elongation: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_mentions = replace_mentions
        self.replace_hashtags = replace_hashtags
        self.keep_emojis = keep_emojis
        self.reduce_elongation = reduce_elongation

        self.char_map = str.maketrans({
            "إ": "ا",
            "أ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ک": "ك",
            "ی": "ي",
            "ھ": "ه",
            "ـ": "",
        })
        self.arabic_diacritics = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
        self.control_chars = re.compile(r"[\u0000-\u001F\u007F-\u009F]")
        self.url_pattern = re.compile(r"(http\S+|www\.\S+)")
        self.email_pattern = re.compile(r"\S+@\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.whitespace_pattern = re.compile(r"\s+")
        self.elongation_pattern = re.compile(r"(.)\1{2,}")
        self.repeated_punct_pattern = re.compile(r"([!?.،؛])\1{2,}")

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = self.control_chars.sub(" ", text)
        text = self.url_pattern.sub(" <URL> ", text) if self.replace_urls else text
        text = self.email_pattern.sub(" <EMAIL> ", text) if self.replace_emails else text
        text = self.mention_pattern.sub(" <USER> ", text) if self.replace_mentions else text
        text = self.hashtag_pattern.sub(r" \1 ", text) if self.replace_hashtags else text
        text = text.translate(self.char_map)
        if self.remove_diacritics:
            text = self.arabic_diacritics.sub("", text)
        if self.reduce_elongation:
            text = self.elongation_pattern.sub(r"\1\1", text)
        text = self.repeated_punct_pattern.sub(r"\1\1", text)
        if not self.keep_emojis and emoji is not None:
            text = emoji.replace_emoji(text, replace=" ")
        return self.whitespace_pattern.sub(" ", text).strip()

    def full_preprocess(self, text: str) -> str:
        return self.clean(text)


preprocessor = ArabicPreprocessor()


# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@torch.no_grad()
def predict_all_aspects(review_text: str) -> Dict[str, Dict[str, Any]]:
    clean_review = preprocessor.full_preprocess(review_text)
    results = {}

    for aspect in ASPECTS:
        prompt = ASPECT_PROMPTS[aspect]
        enc = tokenizer(
            clean_review,
            prompt,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits.detach().cpu().numpy()[0]
        probs = softmax_np(logits)

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        prob_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}

        results[aspect] = {
            "aspect": aspect,
            "pred_label": pred_label,
            "mention_confidence": 1.0 - prob_dict.get("not_mentioned", 0.0),
            "sentiment_confidence": prob_dict.get(pred_label, 0.0),
            "probs": prob_dict,
        }
    return results


def build_output(review_text: str, threshold: float) -> Dict[str, Any]:
    per_aspect = predict_all_aspects(review_text)

    detected_aspects: List[str] = []
    aspect_sentiments: Dict[str, str] = {}
    all_rows = []
    detected_rows = []

    for aspect in ASPECTS:
        info = per_aspect[aspect]
        pred_label = info["pred_label"]
        mention_conf = float(info["mention_confidence"])
        detected = pred_label != "not_mentioned" and mention_conf >= threshold

        row = {
            "Aspect": ASPECT_DISPLAY.get(aspect, aspect),
            "Detected?": "Yes" if detected else "No",
            "Predicted sentiment": SENTIMENT_DISPLAY.get(pred_label, pred_label),
            "Mention confidence": round(mention_conf, 3),
            "Sentiment confidence": round(float(info["sentiment_confidence"]), 3),
        }
        all_rows.append(row)

        if detected:
            detected_aspects.append(aspect)
            aspect_sentiments[aspect] = pred_label
            detected_rows.append(row)

    if not detected_aspects:
        detected_aspects = [OUTPUT_NONE_LABEL]
        aspect_sentiments = {}

    submission_json = {
        "aspects": detected_aspects,
        "aspect_sentiments": aspect_sentiments,
    }

    return {
        "clean_text": preprocessor.full_preprocess(review_text),
        "detected_rows": detected_rows,
        "all_rows": all_rows,
        "submission_json": submission_json,
    }


def render_badges(submission_json: Dict[str, Any]) -> str:
    aspects = submission_json["aspects"]
    sentiments = submission_json["aspect_sentiments"]

    if aspects == [OUTPUT_NONE_LABEL]:
        return "<div class='empty'>No specific aspect was detected. The review may be too short or too general.</div>"

    badges = []
    for aspect in aspects:
        sentiment = sentiments.get(aspect, "neutral")
        css_class = SENTIMENT_CLASS.get(sentiment, "neutral")
        aspect_name = html.escape(ASPECT_DISPLAY.get(aspect, aspect))
        sentiment_name = html.escape(SENTIMENT_DISPLAY.get(sentiment, sentiment))
        badges.append(
            f"<span class='badge {css_class}'><b>{aspect_name}</b> · {sentiment_name}</span>"
        )
    return "<div class='badge-wrap'>" + "".join(badges) + "</div>"


def predict_for_demo(review_text: str, threshold: float):
    if not review_text or not review_text.strip():
        empty_json = {"aspects": [OUTPUT_NONE_LABEL], "aspect_sentiments": {}}
        return (
            "Please enter a review first.",
            [],
            empty_json,
            [],
            "",
        )

    output = build_output(review_text, threshold)
    detected_summary = render_badges(output["submission_json"])
    clean_text = html.escape(output["clean_text"])
    explanation = (
        f"{detected_summary}"
        f"<div class='clean-box'><b>Preprocessed text:</b><br>{clean_text}</div>"
    )

    return (
        explanation,
        output["detected_rows"],
        output["submission_json"],
        output["all_rows"],
        f"Device: {device} · Threshold: {threshold:.2f} · Max tokens: {MAX_LEN}",
    )


DEMO_EXAMPLES = [
    [
        "الخدمة كانت محترمة جدا والمكان رايق، بس الأكل وصل بارد واللحمة ناشفة والسعر مبالغ فيه 😕",
        DEFAULT_THRESHOLD,
    ],
    [
        "The app is smooth and checkout was fast, but el delivery اتأخر ساعتين والدعم مردش عليا خالص.",
        DEFAULT_THRESHOLD,
    ],
    [
        "el akl kan gamed awy 😍 bas el service slow w el bill kan aghla mn el expected.",
        DEFAULT_THRESHOLD,
    ],
    [
        "المكان تحفة والنضافة ممتازة والستاف لطيف، لكن المنيو مفيهوش اختيارات كتير للأطفال.",
        DEFAULT_THRESHOLD,
    ],
    [
        "J'adore l'ambiance et le café était excellent، بس الحمام مش نضيف والخدمة محتاجة تنظيم.",
        DEFAULT_THRESHOLD,
    ],
    [
        "طلبت من التطبيق: الخصم اشتغل، الدفع كان سهل، لكن الأوردر ناقص والدليفري قفل تليفونه.",
        DEFAULT_THRESHOLD,
    ],
    [
        "Honestly wow: clean branch, friendly team, tasty burger, بس السعر عالي شوية مقارنة بالحجم.",
        DEFAULT_THRESHOLD,
    ],
]

CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
.hero {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    border: 1px solid #e5e7eb;
    margin-bottom: 14px;
}
.hero h1 {margin-bottom: 6px;}
.hero p {font-size: 16px; color: #374151;}
.badge-wrap {display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 14px 0;}
.badge {padding: 9px 12px; border-radius: 999px; font-size: 15px; border: 1px solid #e5e7eb;}
.positive {background: #ecfdf5; color: #065f46;}
.negative {background: #fef2f2; color: #991b1b;}
.neutral {background: #fffbeb; color: #92400e;}
.muted {background: #f3f4f6; color: #4b5563;}
.empty {padding: 12px; border-radius: 12px; background: #f3f4f6; color: #4b5563;}
.clean-box {padding: 12px; border-radius: 12px; background: #ffffff; border: 1px solid #e5e7eb; margin-top: 10px;}
.footer-note {font-size: 13px; color: #6b7280;}
"""

with gr.Blocks(css=CSS, title="Arabic ABSA Review Intelligence") as demo:
    gr.HTML(
        """
        <div class='hero'>
            <h1>Review Intelligence: Aspect-Based Sentiment Analysis</h1>
            <p>
                This demo reads a customer review, detects the mentioned aspects
                — food, service, price, cleanliness, delivery, ambiance, app experience, and general opinion —
                then predicts sentiment for each detected aspect.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            review_input = gr.Textbox(
                label="Customer review",
                placeholder="Paste an Arabic, English, French, or mixed Franco-Arabic review here...",
                lines=7,
            )
        with gr.Column(scale=1):
            threshold = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=DEFAULT_THRESHOLD,
                step=0.05,
                label="Aspect detection threshold",
                info="Lower = more aspects detected. Higher = stricter detection.",
            )
            run_btn = gr.Button("Analyze Review", variant="primary")

    status = gr.Markdown(value="")
    summary_html = gr.HTML(label="Executive summary")

    with gr.Tab("Detected aspects"):
        detected_table = gr.Dataframe(
            headers=[
                "Aspect",
                "Detected?",
                "Predicted sentiment",
                "Mention confidence",
                "Sentiment confidence",
            ],
            label="Detected aspect-level sentiment",
            interactive=False,
        )

    with gr.Tab("Submission JSON"):
        output_json = gr.JSON(label="Model output schema")

    with gr.Tab("All aspect scores"):
        all_scores_table = gr.Dataframe(
            headers=[
                "Aspect",
                "Detected?",
                "Predicted sentiment",
                "Mention confidence",
                "Sentiment confidence",
            ],
            label="All aspect decisions",
            interactive=False,
        )

    gr.Examples(
        examples=DEMO_EXAMPLES,
        inputs=[review_input, threshold],
        label="Demo examples",
        cache_examples=False,
    )

    gr.HTML(
        """
        <p class='footer-note'>
        Tip for your LinkedIn demo: start with a simple Arabic review, then use a mixed English/Arabic example,
        then finish with a Franco-Arabic example to show robustness to informal customer language.
        </p>
        """
    )

    run_btn.click(
        fn=predict_for_demo,
        inputs=[review_input, threshold],
        outputs=[summary_html, detected_table, output_json, all_scores_table, status],
    )
    review_input.submit(
        fn=predict_for_demo,
        inputs=[review_input, threshold],
        outputs=[summary_html, detected_table, output_json, all_scores_table, status],
    )

if __name__ == "__main__":
    demo.queue().launch(share=False)
