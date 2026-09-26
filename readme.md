<div align="center">

# 💬 MCA eConsultation — Sentiment Analysis (NLP)

**A leakage-free DistilBERT pipeline for classifying public comments on proposed MCA rules as Positive, Negative, or Neutral**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.56.0-FFD21E?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-blue?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google_Colab-Trained-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Requirements](#-requirements)
3. [Project Structure](#-project-structure)
4. [Data](#-data)
5. [Installation](#-installation)
6. [Project Methodology](#-project-methodology)
7. [Visualization](#-visualization)
8. [Technology Used](#-technology-used)
9. [Future Improvements](#-future-improvements)

---

## 🔎 Overview

This project fine-tunes **DistilBERT** to classify public comments submitted during MCA (Ministry of Corporate Affairs) eConsultation processes into **Positive / Negative / Neutral** sentiment — simulating a real platform use case: given thousands of comments on a proposed rule, how many support it, oppose it, or are undecided?

This version fixes a **data-leakage bug** from an earlier iteration by splitting train/val/test **before** any text augmentation, so validation and test sets are guaranteed to never contain augmented duplicates of training comments.

| Metric (Test Set) | Score |
|---|---|
| 🎯 Accuracy | **1.000** |
| 📐 Precision | **1.000** |
| 🔁 Recall | **1.000** |
| 🧮 F1 Score | **1.000** |
| 📉 Eval Loss | 0.0000250 |

> ⚠️ **Note:** These perfect scores reflect a **synthetic dataset** built for prototyping, not real-world MCA comment data — patterns are more separable than genuine public comments would be. Treat this as a validated pipeline, not a production-ready accuracy claim (see [Future Improvements](#-future-improvements)).

---

## ⚙️ Requirements

- Python 3.10+
- GPU recommended (trained on Google Colab)
- Google Drive (for dataset + checkpoint storage)

**Python packages:**
```
datasets
torch
transformers==4.56.0
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
emoji
accelerate
nlpaug
```

---

## 🗂️ Project Structure

```
mca_econsultation_leakage_free/
├── comments_dataset.csv           # raw input: comment, sentiment
├── train_augmented.csv            # saved augmented training set
└── Sentimental_comments/
    ├── results/                   # Trainer checkpoints (per-epoch)
    ├── logs/                      # training logs
    └── final_model/               # saved fine-tuned model + tokenizer
```

---

## 📊 Data

- **Format:** CSV with two columns — `comment` (raw text) and `sentiment` (positive / negative / neutral)
- **Nature:** Synthetic dataset built for prototyping (not scraped from live MCA submissions)
- **Split:** 80% train / 10% validation / 10% test — **stratified by label**, done *before* augmentation
- **Augmentation:** Applied only to the training split — synonym replacement, random insertion, random swap, random deletion, plus extra keyword-based augmentation for the neutral class (to address class imbalance)

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Install dependencies
pip install -q datasets torch scikit-learn pandas numpy matplotlib seaborn wordcloud nltk emoji accelerate nlpaug
pip install --upgrade transformers==4.56.0

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Place comments_dataset.csv in the expected data folder, then run the script/notebook
```

---

## 🧭 Project Methodology

1. **Mount storage** — Google Drive for dataset and model persistence
2. **Load data** — CSV with `comment` and `sentiment` columns
3. **Modular preprocessing pipeline** — each cleaning technique (emoji removal, special-character stripping, lowercasing) is a standalone function chained in an ordered list, so steps can be added/removed without touching other code
4. **Label mapping** — sentiment strings mapped to integers (negative=0, neutral=1, positive=2)
5. **Split before augmentation** *(the key leakage fix)* — train/val/test split on original, clean comments only, so val/test are never touched by augmentation
6. **Modular augmentation pipeline (train set only)** — synonym replacement, random insertion, random swap, random deletion, plus keyword augmentation for underrepresented neutral comments
7. **Tokenize** — DistilBERT tokenizer, max length 90, padded/truncated
8. **Wrap in PyTorch `Dataset`** — custom `FeedbackDataset` class for train/val/test
9. **Load model** — `distilbert-base-uncased` fine-tuned for 3-class classification
10. **Train** — Hugging Face `Trainer`, 3 epochs, batch size 16, learning rate 3e-5, best model loaded at end
11. **Evaluate on held-out test set** — honest score, no augmented/leaked data
12. **Save model + tokenizer** — for reuse in inference
13. **Predict function** — reuses the same preprocessing pipeline for consistent inference on new text
14. **Full-dataset prediction** — simulates the real deployment use case: sentiment breakdown across all submitted comments
15. **Visualization** — bar chart of sentiment distribution, word clouds (overall + per sentiment)
16. **Summary report** — plain-text breakdown of comment counts, percentages, and top words per sentiment class

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Epochs | 3 |
| Batch size (train/eval) | 16 / 16 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Max sequence length | 90 |
| Eval/save strategy | Per epoch |
| Best model selection | `load_best_model_at_end=True` |

**Training results (per epoch):**

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | 0.0129 | 0.000112 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.0068 | 0.000035 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 0.0026 | 0.000025 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 📈 Visualization

Generated at the end of the pipeline:

- 📊 **Sentiment distribution bar chart** — count of Positive / Neutral / Negative predictions across all comments
- ☁️ **Word clouds** — one for all comments combined, plus one per sentiment class, to surface the most common terms driving each label
- 📝 **Text summary report** — total comments analyzed, per-class percentage breakdown, and top 8 words per sentiment

*(Add your actual chart/word-cloud images here once exported, e.g. `![Sentiment Distribution](results/sentiment_distribution.png)` — happy to wire these in with real filenames, same as the traffic sign project.)*

---

## 🧰 Technology Used

| Category | Tools |
|---|---|
| Model | ![DistilBERT](https://img.shields.io/badge/DistilBERT-base--uncased-blue) |
| Framework | ![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) |
| Text augmentation | ![nlpaug](https://img.shields.io/badge/nlpaug-synonym%2Finsert%2Fswap%2Fdelete-8A2BE2) |
| NLP preprocessing | ![NLTK](https://img.shields.io/badge/NLTK-stopwords%2Ftokenize-green) |
| Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white) ![WordCloud](https://img.shields.io/badge/WordCloud-lightgrey) |
| Experiment tracking | ![W&B](https://img.shields.io/badge/Weights_%26_Biases-offline_mode-FFBE00?logo=weightsandbiases&logoColor=black) |
| Compute | ![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab&logoColor=white) |

---

## 🚀 Future Improvements

- 🧪 **Validate on real, unseen MCA comments** — current perfect scores come from a synthetic dataset; the true test is generalization to genuine public submissions
- ⚖️ Further address class imbalance beyond keyword-based neutral augmentation (e.g. back-translation, LLM-generated paraphrases)
- 🌐 Add support for Hindi / code-mixed comments, common in real MCA submissions
- 📦 Wrap the trained model behind a simple API endpoint for integration into an MCA-facing dashboard
- 📊 Add a confusion matrix and per-class precision/recall breakdown to catch class-specific weaknesses that overall accuracy can hide
- 🔁 Re-run evaluation periodically as new comment batches arrive, to monitor for concept drift

---

<div align="center">
Built with DistilBERT · Trained on Google Colab
</div>
