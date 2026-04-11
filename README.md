# 🎬 IMDB Sentiment Analysis using LSTM

Binary sentiment classification (positive/negative) on 50,000 IMDB 
movie reviews using deep learning.

---

## 📈 Results

| Model                  | Accuracy | F1-Score |
|------------------------|----------|----------|
| Basic LSTM (v1)        | ~86%     | 0.85     |
| BiLSTM + GloVe (v2)   | ~93%     | 0.93     |

> **+7% accuracy improvement** through architectural upgrades 
  and pretrained GloVe embeddings.

---

## 🔧 Improvements from v1 → v2

| Area | v1 (Basic LSTM) | v2 (BiLSTM + GloVe) |
|---|---|---|
| Preprocessing | Basic tokenization | HTML removal + negation handling |
| Vocabulary | 5,000 words | 20,000 words |
| Embeddings | Random init | Pretrained GloVe 100d |
| Architecture | Uni-directional LSTM | Bidirectional LSTM |
| Regularization | Dropout 0.2 | Dropout 0.3 + Early Stopping |
| Evaluation | Accuracy only | Accuracy, F1, Precision, Recall, Confusion Matrix |

---

## 🏗️ Model Architecture (v2)
