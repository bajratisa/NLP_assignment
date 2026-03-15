
import os
import pickle
import torch
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from groq import Groq

# ── Setup ────────────────────────────────────────────────────

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load embedding model ─────────────────────────────────────

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model = model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# ── Load saved vector database ───────────────────────────────

@st.cache_resource
def load_vector_db():
    with open("vector_db_contextual.pkl", "rb") as f:
        return pickle.load(f)

vector_db = load_vector_db()

# ── Helper functions ─────────────────────

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=5):
    query_embedding = get_embedding(query)
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def generate_answer(query, retrieved_knowledge):
    instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chr(10).join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}

Question: {query}
Chatbot response:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": instruction_prompt}],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Chapter 6 Chatbot",
    layout="centered"
)


st.markdown("""
    <style>
        /* dark background */
        .stApp { background-color: #1e1e1e; }
        
        /* chat messages */
        .stChatMessage { background-color: #2d2d2d; border-radius: 8px; }
        
        /* expander */
        .streamlit-expanderHeader { color: #888888; font-size: 0.8em; }
        
        /* source text */
        .source-text { color: #888888; font-size: 0.75em; }
    </style>
""", unsafe_allow_html=True)

# ── UI ────────────────────────────────────────────────────────

st.title("Chapter 6 Chatbot")
st.caption("Neural Networks — Jurafsky & Martin")
st.divider()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("sources"):
                for i, (chunk, similarity) in enumerate(message["sources"]):
                    st.caption(f"[{i+1}] ({similarity:.2f}) {chunk[:200]}...")

# Chat input
if prompt := st.chat_input("Ask something about Chapter 6..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner(""):
            retrieved = retrieve(prompt)
            answer = generate_answer(prompt, retrieved)
            st.markdown(answer)
            with st.expander("sources"):
                for i, (chunk, similarity) in enumerate(retrieved):
                    st.caption(f"[{i+1}] ({similarity:.2f}) {chunk[:200]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })