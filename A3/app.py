import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_dim):
        super().__init__()
        self.method = method
        if method == "additive":
            self.fc_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.fc_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        s = hidden.squeeze(0)
        src_len = encoder_outputs.shape[1]

        s_expanded = s.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(
            self.fc_encoder(encoder_outputs) + self.fc_hidden(s_expanded)
        )
        attention_scores = self.v(energy).transpose(1, 2)

        return F.softmax(attention_scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout, attention):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hidden_dim + hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs)
        context = torch.bmm(a, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        prediction = self.fc_out(
            torch.cat((output, context, embedded), dim=1)
        )

        return prediction, hidden, a


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


# ==========================================
# 2. LOAD MODEL & VOCABS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocab:
    def __init__(self):
        pass

with open("src_vocab.pkl", "rb") as f:
    src_vocab = pickle.load(f)

with open("trg_vocab.pkl", "rb") as f:
    trg_vocab = pickle.load(f)

INPUT_DIM = src_vocab.num_words
OUTPUT_DIM = trg_vocab.num_words
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
DROPOUT = 0.5

attn = Attention("additive", HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(
    torch.load("Additive_Attention-model.pt", map_location=device)
)
model.eval()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def numericalize(sentence, vocab):
    tokens = sentence.split()
    indices = [
        vocab.word2idx.get(word, vocab.word2idx["<unk>"])
        for word in tokens
    ]
    return (
        [vocab.word2idx["<sos>"]]
        + indices
        + [vocab.word2idx["<eos>"]]
    )


def translate(sentence):
    tokens = numericalize(sentence.lower(), src_vocab)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

        trg_indices = [trg_vocab.word2idx["<sos>"]]
        attention_matrix = []

        for _ in range(50):
            trg_tensor = torch.LongTensor(
                [trg_indices[-1]]
            ).to(device)

            prediction, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs
            )

            attention_matrix.append(
                attention.squeeze(0).cpu().numpy()
            )

            pred_token = prediction.argmax(1).item()
            trg_indices.append(pred_token)

            if pred_token == trg_vocab.word2idx["<eos>"]:
                break

    trg_words = [trg_vocab.idx2word[i] for i in trg_indices]
    return trg_words[1:-1], attention_matrix[:-1]


# ==========================================
# 4. DASH APP
# ==========================================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={
        "maxWidth": "800px",
        "margin": "40px auto",
        "fontFamily": "Arial"
    },
    children=[
        html.H2(
            "A3: English to Nepali Translator",
            style={"textAlign": "center"}
        ),

        dcc.Textarea(
            id="input-text",
            placeholder="Enter English sentence here...",
            style={
                "width": "100%",
                "height": "100px",
                "fontSize": "16px",
                "padding": "10px"
            }
        ),

        html.Button(
            "Translate",
            id="translate-btn",
            n_clicks=0,
            style={
                "marginTop": "15px",
                "padding": "10px 20px",
                "fontSize": "16px"
            }
        ),

        html.H4("Translation:", style={"marginTop": "30px"}),
        html.Div(
            id="output-text",
            style={"fontSize": "18px", "color": "blue"}
        ),

        dcc.Graph(
            id="attention-heatmap",
            config={"displayModeBar": False}
        )
    ]
)

# ==========================================
# 5. CALLBACK
# ==========================================
@app.callback(
    [Output("output-text", "children"),
     Output("attention-heatmap", "figure")],
    Input("translate-btn", "n_clicks"),
    State("input-text", "value")
)
def run_translation(n_clicks, text):
    if not text:
        return "", {}

    translated_words, attn_matrix = translate(text)
    translation = " ".join(translated_words)

    if len(attn_matrix) == 0:
        return translation, {}

    attn_data = np.concatenate(attn_matrix, axis=0)

    fig = px.imshow(
        attn_data,
        labels=dict(
            x="Source (English)",
            y="Target (Nepali)",
            color="Attention"
        ),
        x=["<sos>"] + text.split() + ["<eos>"],
        y=translated_words,
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        title="Attention Heatmap",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return translation, fig


if __name__ == "__main__":
    app.run(debug=True)
    
