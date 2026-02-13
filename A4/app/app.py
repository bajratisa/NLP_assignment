
import dash
from dash import dcc, html, Input, Output, State
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertTokenizerFast

# Model Architecture 
class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments=2, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(embedding))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / math.sqrt(self.d_k)
        if attn_mask is not None:
             scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.layernorm(self.fc(context) + Q), attn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layernorm(self.fc2(self.dropout(F.gelu(self.fc1(x)))) + x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask):
        x, _ = self.attn(x, x, x, attn_mask)
        return self.ffn(x), _

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, n_segments, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, n_segments, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
    def forward(self, input_ids, attention_mask, segment_ids=None):
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        output = self.embedding(input_ids, segment_ids)
        for layer in self.layers:
            output, _ = layer(output, attention_mask)
        return output

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

class SiameseNLI(nn.Module):
    def __init__(self, pretrained_bert, d_model, num_classes=3):
        super().__init__()
        self.bert = pretrained_bert
        self.pooler = MeanPooling()
        self.classifier = nn.Linear(3 * d_model, num_classes)

    def get_sentence_embedding(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        return self.pooler(output, attention_mask)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        u = self.get_sentence_embedding(input_ids_a, attention_mask_a)
        v = self.get_sentence_embedding(input_ids_b, attention_mask_b)
        uv_abs = torch.abs(u - v)
        features = torch.cat([u, v, uv_abs], dim=1)
        logits = self.classifier(features)
        return logits, u, v

# 2. Global Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128

# Rebuild architecture & load weights
base_bert = BERT(vocab_size=30522, d_model=768, n_layers=6, n_heads=8, d_ff=3072, max_len=128, n_segments=2, dropout=0.1)
model = SiameseNLI(base_bert, d_model=768, num_classes=3)

try:
    
    model.load_state_dict(torch.load("../sbert_snli_scratch.pth", map_location=device))
    print(" Weights loaded successfully.")
except FileNotFoundError:
    print(" Weights file not found! Please ensure 'sbert_snli_scratch.pth' is in the root directory.")
    
model.to(device)
model.eval()
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 3. Dash App Initialization & Layout 
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'}, children=[
    html.H1("A4: Do you AGREE?", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.P("Determine whether a Hypothesis is an Entailment, Neutral, or Contradiction to the Premise.", style={'textAlign': 'center'}),
    html.Hr(),
    
    html.Div([
        html.Label("Premise:", style={'fontWeight': 'bold'}),
        dcc.Textarea(
            id='premise-input',
            value="",
            style={'width': '100%', 'height': '80px', 'marginBottom': '15px', 'padding': '10px'}
        ),
        
        html.Label("Hypothesis:", style={'fontWeight': 'bold'}),
        dcc.Textarea(
            id='hypothesis-input',
            value="",
            style={'width': '100%', 'height': '80px', 'marginBottom': '15px', 'padding': '10px'}
        ),
        
        html.Button(
            'Predict', 
            id='predict-button', 
            n_clicks=0, 
            style={'backgroundColor': '#2980b9', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '16px'}
        ),
    ]),
    
    html.Hr(),
    html.Div(id='output-container', style={'marginTop': '20px'})
])

# 4. Callbacks (Inference Logic)
@app.callback(
    Output('output-container', 'children'),
    Input('predict-button', 'n_clicks'),
    State('premise-input', 'value'),
    State('hypothesis-input', 'value')
)
def update_output(n_clicks, premise, hypothesis):
    if n_clicks > 0:
        if not premise or not hypothesis:
            return html.Div(" Please enter both a premise and a hypothesis.", style={'color': 'red'})

        # Tokenize
        enc_a = tokenizer(premise, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        enc_b = tokenizer(hypothesis, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids_a = enc_a['input_ids'].to(device)
        mask_a = enc_a['attention_mask'].to(device)
        input_ids_b = enc_b['input_ids'].to(device)
        mask_b = enc_b['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            logits, u, v = model(input_ids_a, mask_a, input_ids_b, mask_b)
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            prediction = torch.argmax(logits, dim=1).item()
            cos_sim = F.cosine_similarity(u, v).item()
            
        labels = ['Entailment', 'Neutral', 'Contradiction']
        predicted_label = labels[prediction]
        
        # Return formatted HTML
        return html.Div([
            html.H3("Results"),
            html.Div([
                html.Strong("Prediction: "), html.Span(predicted_label, style={'color': '#27ae60', 'fontWeight': 'bold', 'fontSize': '18px'})
            ], style={'backgroundColor': '#e8f8f5', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '15px'}),
            
            html.P([html.Strong("Cosine Similarity: "), f"{cos_sim:.4f}"]),
            
            html.Strong("Confidence Scores:"),
            html.Ul([
                html.Li(f"Entailment: {probs[0]:.2%}"),
                html.Li(f"Neutral: {probs[1]:.2%}"),
                html.Li(f"Contradiction: {probs[2]:.2%}")
            ])
        ])
    return ""

if __name__ == '__main__':
    app.run(debug=True)
