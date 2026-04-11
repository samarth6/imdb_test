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

##  Improvements from v1 → v2

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

```
Embedding (GloVe 100d — frozen)
          ↓
Bidirectional LSTM (128 units, dropout=0.3)
          ↓
GlobalMaxPooling1D
          ↓
Dense (64 units, ReLU) + Dropout(0.3)
          ↓
Dense (1 unit, Sigmoid)
```

--- 

 How to Run

1. Clone the repo
```bash
git clone https://github.com/samarth6/IMDB_sentiment_analysis
cd IMDB_sentiment_analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open notebook in Google Colab  
   - v1 Basic: `notebooks/v1_basic_lstm.ipynb`  
   - v2 Improved: `notebooks/v2_bilstm_glove.ipynb`

---

##  Tech Stack

`Python` `TensorFlow` `Keras` `GloVe` `NumPy` `Pandas` `Scikit-learn` `Matplotlib` `Seaborn`

---

##  Project Structure

```
│
├── notebooks/
│   ├── v1_basic_lstm.ipynb        # Baseline LSTM model
│   └── v2_bilstm_glove.ipynb      # Improved BiLSTM + GloVe model
│
├── results/
│   ├── training_history.png       # Accuracy & loss curves
│   └── confusion_matrix.png       # Model evaluation heatmap
│
├── .gitignore
├── requirements.txt
└── README.md
```
