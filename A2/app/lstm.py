import torch
import torch.nn as nn
import math

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        # 1. Embedding layer: Turns word IDs into rich vectors of numbers 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # 2. LSTM layer: The actual brain that remembers context 
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, 
                            dropout=dropout_rate, batch_first=True)
        
        # 3. Dropout: Randomly turns off some neurons to prevent "overfitting" (cheating by memorizing) 
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. FC (Linear) layer: Turns the brain's thoughts back into word predictions 
        self.fc = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights with small random numbers so the model starts learning fresh 
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def init_hidden(self, batch_size, device):
        # Create the initial "empty memory" for the LSTM 
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        # This prevents the memory from getting too heavy for the computer to handle 
        hidden, cell = hidden
        return hidden.detach(), cell.detach()

    def forward(self, src, hidden):
        # src is the input sequence of word IDs 
        # embedding: [batch size, seq len, emb_dim] 
        embed = self.dropout(self.embedding(src))
        
        # output: [batch size, seq len, hid_dim] 
        output, hidden = self.lstm(embed, hidden)      
        
        output = self.dropout(output) 
        # prediction: [batch size, seq len, vocab size] 
        prediction = self.fc(output)
        
        return prediction, hidden