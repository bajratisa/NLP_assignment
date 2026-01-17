from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

import pickle

# Load embeddings
with open("./model/skipgram_embeddings.pkl", "rb") as f:
    skipgram_embeddings_dict = pickle.load(f)

with open("./model/neg_embeddings.pkl", "rb") as f:
    neg_embeddings_dict = pickle.load(f)

with open("./model/glove_embeddings.pkl", "rb") as f:
    glove_embeddings_dict = pickle.load(f)

# 🔹 Register models here
MODELS = {
    "skipgram": skipgram_embeddings_dict,
    "neg": neg_embeddings_dict,
    "glove": glove_embeddings_dict
}

def top_k_similar(query_word, embeddings_dict, k=10):
    if query_word not in embeddings_dict:
        return None, f"'{query_word}' not in vocabulary"

    query_vec = embeddings_dict[query_word]
    scores = []

    for word, vec in embeddings_dict.items():
        if word != query_word:
            score = float(np.dot(query_vec, vec))
            scores.append((word, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k], None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/similar", methods=["GET"])
def get_similar_words():
    word = request.args.get("word", "").lower()
    model_name = request.args.get("model", "").lower()

    if not word or not model_name:
        return jsonify({
            "error": "Parameters 'word' and 'model' are required"
        }), 400

    if model_name not in MODELS:
        return jsonify({
            "error": f"Unknown model '{model_name}'. Choose from {list(MODELS.keys())}"
        }), 400

    embeddings = MODELS[model_name]
    results, error = top_k_similar(word, embeddings)

    if error:
        return jsonify({"error": error}), 404

    return jsonify({
        "query": word,
        "model": model_name,
        "top_10": [
            {"word": w, "score": round(s, 4)}
            for w, s in results
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
