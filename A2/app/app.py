from dash import Dash, html, dcc, Input, Output, State
import torch
import pickle
import os
import sys
import re

# Add the current directory to the path to locate lstm.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from lstm import LSTMLanguageModel

# --- 1. Vocabulary Class Definition ---
# Pickle needs this class defined to load the vocab_lm.pkl file
class Vocabulary:
    def __init__(self, vocab_dict, unk_idx):
        self.vocab_dict = vocab_dict
        self.unk_idx = unk_idx
        self.itos = {v: k for k, v in vocab_dict.items()}
        
    def __getitem__(self, token):
        return self.vocab_dict.get(token, self.unk_idx)
    
    def __len__(self):
        return len(self.vocab_dict)
    
    def get_itos(self):
        return [self.itos[i] for i in range(len(self))]

# --- 2. Setup and Model Loading ---
device = torch.device("cpu")

# Load the Vocabulary
vocab_path = os.path.join(current_dir, "../models/vocab_lm.pkl")
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Load the Trained Model weights
model_path = os.path.join(current_dir, "../models/best-val-lstm_lm.pt")

# Hyperparameters matching the training notebook exactly
vocab_size = len(vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 3. Text Generation Logic ---
def generate_text(prompt, max_seq_len, temperature, model, vocab, device):
    # Basic tokenizer to split the prompt into words
    tokens = re.findall(r"\w+|[^\w\s]", prompt.lower())
    indices = [vocab[t] for t in tokens]
    
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # Use temperature to adjust prediction randomness
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            if prediction == vocab['<eos>']: 
                break
            indices.append(prediction)

    itos = vocab.get_itos()
    return " ".join([itos[i] for i in indices])

# --- 4. Dash App Configuration ---
app = Dash(__name__)

# CSS to hide the Dash debug bar and clean up the interface
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>A2 Language Model</title>
        {%favicon%}
        {%css%}
        <style>
            .dash-debug-menu, .dash-renderer-error-wrap { display: none !important; }
            body { background-color: #ffffff; font-family: 'Segoe UI', Arial, sans-serif; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# --- 5. Layout ---
app.layout = html.Div([
    html.H2("A2 Language Model", style={'textAlign': 'center', 'marginTop': '60px', 'color': '#333'}),
    
    html.Div([
        dcc.Input(
            id='input-text', 
            type='text', 
            placeholder='Enter a prompt...', 
            style={'width': '50%', 'padding': '12px', 'borderRadius': '4px', 'border': '1px solid #ccc', 'fontSize': '16px'}
        ),
        html.Button(
            'Generate', 
            id='submit-button', 
            n_clicks=0, 
            style={'padding': '12px 25px', 'marginLeft': '10px', 'cursor': 'pointer', 'fontSize': '16px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ccc', 'borderRadius': '4px'}
        ),
    ], style={'textAlign': 'center', 'marginTop': '25px'}),
    
    # A single results card to hold all 10 generations
    html.Div(
        id='output-card', 
        style={
            'marginTop': '40px', 
            'maxWidth': '750px', 
            'margin': '40px auto', 
            'padding': '30px', 
            'border': '1px solid #eaeaea', 
            'borderRadius': '10px',
            'backgroundColor': '#fff',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.05)',
            'display': 'none' # Hidden until content is generated
        }
    )
])

# --- 6. Callback ---
@app.callback(
    [Output('output-card', 'children'), Output('output-card', 'style')],
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def update_output(n_clicks, input_value):
    # Default style for the card
    card_style = {
        'marginTop': '40px', 'maxWidth': '750px', 'margin': '40px auto', 
        'padding': '30px', 'border': '1px solid #eaeaea', 'borderRadius': '10px',
        'backgroundColor': '#fff', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)'
    }
    
    if n_clicks > 0:
        if not input_value:
            card_style['display'] = 'block'
            return html.Div("Please enter a prompt.", style={'color': 'red'}), card_style
        
        # 10 different temperatures to ensure diverse results
        temps = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
        
        all_results = [html.H3("Generated Results", style={'borderBottom': '2px solid #eee', 'paddingBottom': '10px', 'marginBottom': '20px'})]
        
        # Loop 10 times to populate the single card
        for i in range(10):
            text = generate_text(input_value, 25, temps[i], model, vocab, device)
            all_results.append(html.Div([
                html.Small(f"Variation {i+1} (Temp: {temps[i]})", style={'color': '#888'}),
                html.P(text, style={'fontSize': '16px', 'marginBottom': '15px', 'lineHeight': '1.5'})
            ], style={'marginBottom': '20px', 'borderBottom': '1px solid #f9f9f9'}))
            
        card_style['display'] = 'block'
        return all_results, card_style
    
    return "", {'display': 'none'}

if __name__ == '__main__':
    # Use modern app.run instead of legacy run_server
    app.run(debug=True)