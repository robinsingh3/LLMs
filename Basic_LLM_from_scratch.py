import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample text data (feel free to replace this with a larger dataset)
text = "hello world"

# Create a character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Convert text to indices
data = [char_to_idx[c] for c in text]

# Hyperparameters
embedding_size = 32
nhead = 2
hidden_size = 64
num_layers = 2
dropout = 0.2
num_epochs = 1000
learning_rate = 0.01
seq_length = 5

# Prepare the dataset
def get_batch(data, seq_length):
    inputs = []
    targets = []
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i + seq_length])
        targets.append(data[i + 1:i + seq_length + 1])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

inputs, targets = get_batch(data, seq_length)
inputs, targets = inputs.to(device), targets.to(device)

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer-based language model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, hidden_size, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        self.transformer = nn.Transformer(embedding_size, nhead, num_layers, num_layers, hidden_size, dropout)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.embedding_size = embedding_size

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)
        output = self.fc_out(output)
        return output

# Instantiate the model, loss function, and optimizer
model = TransformerModel(vocab_size, embedding_size, nhead, hidden_size, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(inputs.transpose(0, 1))
    loss = criterion(output.view(-1, vocab_size), targets.transpose(0, 1).reshape(-1))
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to generate text using the trained model
def generate_text(model, start_str, length):
    model.eval()
    generated = [char_to_idx[ch] for ch in start_str]
    input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(device)

    for _ in range(length):
        output = model(input_seq)
        last_logits = output[-1, 0, :]
        probs = F.softmax(last_logits, dim=0)
        next_char_idx = torch.multinomial(probs, 1).item()
        generated.append(next_char_idx)
        input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=0)

    return ''.join([idx_to_char[idx] for idx in generated])

# Generate text starting with the character 'h'
start_str = "h"
generated_text = generate_text(model, start_str, 20)
print("Generated text:", generated_text)
