# 🎬 IMDB Movie Review Sentiment Classifier

A web application for classifying the sentiment of movie reviews (Positive/Negative) using GRU-based deep learning models, trained on the IMDB Movie Review dataset and deployed with Streamlit.

## 🚀 Live Demo

🔗 [https://imbdfinal.streamlit.app/]([https://imbdfinal.streamlit.app/])

## 📋 Features

- **Dual Model Support** — switch between two GRU model configurations (Config 1 and Config 2)
- **Real-Time Sentiment Prediction** — type any English movie review and get instant sentiment classification
- **Confidence Score** — probability indicator showing model certainty for each prediction
- **Custom Text Preprocessing** — cleans and tokenizes input text to match training conditions

## 🏗️ Project Structure
IMDB_Final/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── GRU_Config1_SeqB.h5 # GRU Model Configuration 1
├── best_model_imdb_GRU_Config2_SeqB.h5 # GRU Model Configuration 2 (Best Model)
└── tokenizer.pickle # Fitted Keras Tokenizer
## 🧠 Model Architecture

Both models use GRU (Gated Recurrent Unit) layers to process sequential text data, differing in their internal configuration (layer size/depth).

**Sequence length:** 250 tokens (SeqB configuration)
**Vocabulary size:** 10,000 most frequent words
**Output:** Binary classification (sigmoid activation) — Positive or Negative

## ⚙️ Text Preprocessing Pipeline
Raw review text
→ Lowercase conversion
→ HTML tag removal (e.g., <br/>)
→ Non-alphabetic character removal
→ Whitespace normalization
→ Tokenization (Keras Tokenizer, fitted on training data)
→ Padding/truncating to fixed sequence length (250)


The tokenizer was fitted directly on the training corpus and saved separately to ensure consistent word-to-index mapping between training and inference.

## 🛠️ Installation & Local Setup

```bash
git clone https://github.com/username/imdb-sentiment-classifier.git
cd imdb-sentiment-classifier
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Dependencies
streamlit==1.62.0
tensorflow==2.21.0
numpy==2.4.2


## 📊 Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| GRU Config 1 | *[isi sesuai hasil training]* | Sequence length: 250 |
| GRU Config 2 (Best) | *[isi sesuai hasil training]* | Sequence length: 250 |

## 📝 Version History

| Version | Changes |
|---|---|
| v1 | Basic prediction — single model, plain text output |
| v2 (Final) | Added model selection dropdown, confidence score display |

## 👤 Author

Built as part of the Learning Advance Skills (LAS) 2026 program at MBC Laboratory, Big Data Division.

## 📄 License

This project was developed for educational purposes as part of the LAS 2026 program.
