# Natural Language Processing Assignments

Coursework for AT82.xx Natural Language Processing at the Asian Institute of Technology, [Semester, Year].

Seven assignments that go from word embeddings to LLM agents. Most of the models are built from scratch in PyTorch, and each assignment comes with a small web app for trying it out.

| # | Project | What it does | Tools |
|---|---|---|---|
| A1 | Word Similarity | Trains Skip-gram, Skip-gram with negative sampling, and GloVe embeddings on the Reuters corpus, evaluates them on WordSim353, and returns the 10 most similar words for any input. | PyTorch, Flask |
| A2 | LSTM Language Model | A 2-layer LSTM trained on Harry Potter and the Sorcerer's Stone that completes a prompt, showing 10 outputs at different sampling temperatures. | PyTorch, Dash |
| A3 | English to Nepali Translation | Seq2Seq encoder-decoder comparing general and additive attention, trained on a 208k sentence English-Nepali corpus, with attention heatmaps in the app. | PyTorch, Dash, Hugging Face Datasets |
| A4 | BERT from Scratch for NLI | Pretrains BERT on WikiText, then adapts it into Sentence-BERT to classify sentence pairs as entailment, neutral or contradiction (63% accuracy on SNLI). | PyTorch, Dash, scikit-learn |
| A5 | DPO Alignment | Fine-tunes Qwen2.5-0.5B-Instruct with DPO and QLoRA for truthfulness, then compares it against the base model using Gemini as a judge. [Model on Hugging Face](https://huggingface.co/Tisa99/Assignment5-Qwen-DPO) | TRL, QLoRA, Gemini API |
| A6 | Naive RAG vs Contextual Retrieval | A QA chatbot over Chapter 6 of Jurafsky and Martin, comparing two retrieval strategies using ROUGE. | BGE embeddings, Groq (Llama 3.1), Streamlit |
| A7 | MCP AI Agent | A Telegram bot agent built in n8n that uses MCP tools and Google Calendar, running locally in Docker. | n8n, Docker, PostgreSQL, Groq |

Each folder has its own README with setup steps and results.
