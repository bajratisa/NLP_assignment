# Word Similarity API

A web application and API to find the **top 10 most similar words** for a given input word using different word embedding models: Skip-gram, Skip-gram with Negative Sampling (NEG), and GloVe.

---

## Datasets Used

- **Reuters Corpus** (for training embeddings)  
  [NLTK Book - Chapter 2](https://www.nltk.org/book/ch02.html)

- **WordSim353 Crowd-sourced** (for similarity evaluation)  
  [Kaggle Dataset](https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd?resource=download)

---

## Requirements

- Python 3.9+
- Flask
- NumPy
- Pandas
- PyTorch
- Pickle

Install dependencies:

```bash
pip install flask numpy pandas torch


NLP-word-similarity/
│
├─ app.py
├─ templates/
│   └─ index.html
├─ model/
│   ├─ skipgram_embeddings.pkl
│   ├─ neg_embeddings.pkl
│   └─ glove_embeddings.pkl
│─ wordsim353crowd.csv
│─ capital-common-countries.txt
│─ past-tense.txt
├─ README.md
```
Submitted By:
Tisa Bajracharya
st126686