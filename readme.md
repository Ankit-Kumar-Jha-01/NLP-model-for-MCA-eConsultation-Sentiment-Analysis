<!-- <div align="center"> -->

# NLP-model-for-MCA-eConsultation-Sentiment-Analysis

<p align="center">A sentiment classification system built to analyze public comments submitted on the MCA (Ministry of Corporate Affairs) eConsultation platform, where citizens comment on proposed rules and regulations. The model classifies each comment as **Positive**, **Negative**, or **Neutral**, and generates a word cloud + summary report to help visualize public opinion at a glance.

Originally built as a hackathon prototype.

## Problem

When a new rule or regulation is proposed, the MCA platform receives large volumes of public comments. Reading through all of them manually to gauge public sentiment isn't practical at scale. This project automates that — given a batch of comments, it predicts the overall sentiment split and surfaces the most common themes.

## Approach

- Fine-tuned **DistilBERT** for 3-class sentiment classification (positive / negative / neutral)
- Built a modular **text preprocessing pipeline** (emoji removal, special character cleanup, normalization) — implemented as a chain of independent steps so techniques can be added or removed without touching other code
- Used **data augmentation** (synonym replacement, random insertion, swap, deletion) to expand the training set, with a stratified train/val/test split performed *before* augmentation to prevent data leakage between splits
- Generated a **word cloud** and a short **summary report** (sentiment breakdown + most common words per class) from model predictions

## Dataset

Due to the private, government-hosted nature of real MCA eConsultation data, this prototype uses a synthetic dataset (5,000+ comments) generated with an LLM and labeled by sentiment, meant to simulate the structure of real platform comments for the hackathon build.

## Tech Stack

Python, PyTorch, Hugging Face Transformers (DistilBERT), scikit-learn, nlpaug, pandas, WordCloud, NLTK

## Results & Honest Limitations

The model reaches ~99–100% accuracy on the held-out test set. This is expected given the synthetic dataset — LLM-generated comments per sentiment class tend to follow clean, repeated patterns, making the classification task easier than it would be on real, messy human writing.

To sanity-check this, I tested the model on hand-written, natural-sounding sentences outside the dataset (hedged opinions, idioms, mixed-tone comments). It handled clearly positive/negative cases well but misclassified idiomatic phrasing — e.g. it labeled *"no complaints so far"* as Negative, latching onto the word "complaints" without recognizing the negation flips the meaning. This points to the model learning surface-level word patterns from the synthetic data rather than deeper language understanding - a known limitation that real, human-labeled training data would help address.

## What I'd Improve With More Time

- Train on real MCA comments (or a more varied, noisy synthetic set with sarcasm/hedging/negation examples)
- Add negation-aware preprocessing or a rule-based override layer for common idioms
- Expand evaluation with a dedicated "hard examples" test set, separate from the main test split.</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

![Python](https://img.shields.io/badge/-Python-555?style=flat-square&logo=python) ![PyTorch](https://img.shields.io/badge/-PyTorch-555?style=flat-square&logo=pytorch) ![Hugging Face Transformers](https://img.shields.io/badge/-Hugging%20Face%20Transformers-555?style=flat-square&logo=hugging%20face%20transformers) ![DistilBERT](https://img.shields.io/badge/-DistilBERT-555?style=flat-square&logo=distilbert) ![scikit-learn](https://img.shields.io/badge/-scikit-learn-555?style=flat-square&logo=scikit-learn) ![nlpaug](https://img.shields.io/badge/-nlpaug-555?style=flat-square&logo=nlpaug) ![pandas](https://img.shields.io/badge/-pandas-555?style=flat-square&logo=pandas) ![NumPy](https://img.shields.io/badge/-NumPy-555?style=flat-square&logo=numpy) ![NLTK](https://img.shields.io/badge/-NLTK-555?style=flat-square&logo=nltk) ![WordCloud](https://img.shields.io/badge/-WordCloud-555?style=flat-square&logo=wordcloud) ![Matplotlib](https://img.shields.io/badge/-Matplotlib-555?style=flat-square&logo=matplotlib) ![Google Colab](https://img.shields.io/badge/-Google%20Colab-555?style=flat-square&logo=google%20colab)

[🐛 Report Bug](https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis/issues) · [✨ Request Feature](https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis/issues)

<!-- </div> -->

---

## 📋 Table of Contents

- [📸 Screenshots](#screenshots)
- [⚙️ Prerequisites](#prerequisites)
- [🚀 Installation](#installation)
- [💻 Usage](#usage)
- [✨ Features](#features)
- [🗺️ Roadmap](#roadmap)
- [🤝 Contributing](#contributing)
- [❓ FAQ](#faq)
- [📄 License](#license)
- [👤 Contact](#contact)
- [🙏 Acknowledgements](#acknowledgements)

## 📸 Screenshots

> Add your screenshots here.

![Screenshot](./screenshots/screenshot.png)

## ⚙️ Prerequisites

- Python 3.8 or higher
- Google Colab or Jupyter Notebook with GPU support (recommended)
- Google Drive access (if running on Colab, for saving/loading model and dataset)

## 🚀 Installation

```bash
pip install torch transformers scikit-learn pandas numpy nltk wordcloud matplotlib nlpaug emoji accelerate
```

## 💻 Usage

```bash
from predict import predict_sentiment

comment = "The new compliance rule is confusing and adds too much paperwork."
result = predict_sentiment(comment)
print(result)  # Output: Negative
```

## ✨ Features

- ✅ Classifies public comments into Positive, Negative, or Neutral sentiment
- ✅ Modular preprocessing pipeline — add or remove cleaning steps without touching other code
- ✅ Data augmentation (synonym replacement, insertion, swap, deletion) applied only to training data to avoid leakage
- ✅ Stratified train/val/test split performed before augmentation for honest evaluation
- ✅ Generates a word cloud for overall comments and for each sentiment class
- ✅ Produces a short summary report with sentiment breakdown and top keywords per class
- ✅ Reusable predict_sentiment() function for quick testing on new comments

## 🗺️ Roadmap

- [ ] Train on real MCA eConsultation comments instead of synthetic data
- [ ] Add negation-aware preprocessing to fix idiom misclassification (e.g. "no complaints so far")
- [ ] Build a dedicated hard-examples test set for evaluating natural, mixed-tone language
- [ ] Deploy as an API for real-time comment scoring

See the [open issues](https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis/issues) for proposed features and known issues.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ❓ FAQ

**Q: How do I get started?**
A: Follow the installation guide above.

**Q: How do I report a bug?**
A: Open an issue on the [GitHub Issues](https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis/issues) page.

**Q: Can I contribute?**
A: Yes! See the Contributing section above.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Contact

**Ankit Kumar Jha**
- GitHub: [@Ankit-Kumar-Jha-01](https://github.com/Ankit-Kumar-Jha-01)
- Email: [anjha8409@gmail.com](mailto:anjha8409@gmail.com)
- Project: [https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis](https://github.com/Ankit-Kumar-Jha-01/nlp-model-for-mca-econsultation-sentiment-analysis)

## 🙏 Acknowledgements

- Hugging Face for the DistilBERT model and Transformers library
- Ministry of Corporate Affairs (MCA) eConsultation platform for the use-case inspiration

---

<div align="center">Made with ❤️ by Ankit Kumar Jha</div>
