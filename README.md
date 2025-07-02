# Multilingual_translator_ai
This multilingual translator uses Meta’s NLLB-200 1.3B model with Hugging Face Transformers, PyTorch, and NLTK to translate text between 200+ languages. It supports GPU-accelerated inference and sentence-wise translation. Achieved BLEU and CHRF scores above 80, indicating high translation accuracy.
# 🌍 Multilingual Translator using NLLB-200 1.3B

This project is a multilingual text translator powered by Meta’s [NLLB-200 (No Language Left Behind)](https://ai.facebook.com/research/no-language-left-behind/) 1.3B model. It enables dynamic translation between 200+ languages with high accuracy, using sentence-wise processing and GPU acceleration.

---

## 🚀 Features

- 🔄 Translate text between over 200 languages
- 🧠 Built using `facebook/nllb-200-1.3B` model
- 📥 Supports long paragraphs by sentence segmentation
- ⚙️ Preprocessing & Postprocessing for clean output
- 📊 BLEU and CHRF evaluation (scores > 80)
- ⚡ Fast inference with GPU (T4 on Google Colab)

---

## 🛠️ Technologies Used

| Technology      | Purpose                                 |
|-----------------|------------------------------------------|
| Transformers    | Load & run NLLB model                   |
| PyTorch         | Model execution on GPU                  |
| NLTK            | Sentence segmentation                   |
| Regex & Unicode | Text preprocessing & postprocessing     |
| Evaluate        | Compute BLEU and CHRF translation scores|
| SacreBLEU       | Backend for CHRF metric                 |
| Google Colab    | Execution environment with GPU support  |

---

## 📦 Installation

Run the following in a Colab notebook:

```bash
!pip install transformers evaluate sacrebleu
import nltk
nltk.download('punkt')
