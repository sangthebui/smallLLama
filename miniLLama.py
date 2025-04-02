import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


# Define a custom dataset class for the dataset
class WikiDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = self.tokenize_data(dataset["text"])

    def tokenize_data(self, texts):
        tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text, truncation=False)  # Get full tokenized text
            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                tokenized_texts.append(torch.tensor(tokens[i:i + self.seq_len + 1], dtype=torch.long))
        return tokenized_texts
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # Input and target shifted by 1


# Define the simplified LLama architecture
class MiniLLama(nn.Module):
    def __init__(self, vocab_size=50257, d_model=96, num_heads=2, num_layers=2, seq_len=64): 
        super(MiniLLama, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=192, dropout=0.1)  # Reduced feedforward dim
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory):
        x = self.embedding(x)
        memory = self.embedding(memory)  # (batch_size, seq_len, d_model)
        
        # Convert (batch_size, seq_len, d_model) to (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        
        
        output = self.transformer(x, memory)
        output = self.fc_out(output.permute(1, 0, 2))  # Convert back to (batch_size, seq_len, vocab_size)

        return output


# Load the dataset from Hugging Face
dataset = load_dataset('mikasenghaas/wikitext-2', split='train')

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set hyperparameters
vocab_size = tokenizer.vocab_size
hidden_size = 256
num_heads = 4
num_layers = 4
seq_len = 128
batch_size = 32
num_epochs = 10


# Initialize Model and Training Components
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MiniLLama(vocab_size, hidden_size, num_heads, num_layers).to(device)
dataset = WikiDataset(dataset, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Increased batch size from 8 to 16
optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))  # Switched to AdamW for faster convergence
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


# Training Loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, inputs)  # Decoder-only: using inputs as memory
        loss = loss_fn(outputs.view(-1, 50257), targets.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Save the  model
torch.save(model.state_dict(), 'small_llama_chat_model.pth')
print('Model saved to small_llama_chat_model.pth')

# Text Generation Function
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=10):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    generated = input_ids  # (batch_size, seq_len)
    
    for _ in range(max_length):
        with torch.no_grad():
            # Get the logits for the last token in the sequence
            logits = model(generated, generated)[:, -1, :] / temperature  # (batch_size, vocab_size)
            
            # Apply top-k sampling
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)  # (batch_size, top_k)
            random_index = torch.randint(0, top_k, (1,)).item()  # Randomly select an index from top-k
            chosen_index = top_indices[:, random_index].unsqueeze(-1)  # Add sequence length dimension
            
            
        # Concatenate the chosen token to the generated sequence
        generated = torch.cat([generated, chosen_index], dim=1)  # Concatenate along sequence length dimension
    
    results = generated[0].tolist()
    answer = tokenizer.decode(results, skip_special_tokens=True)
    return answer

# Load the fine-tuned model
model.load_state_dict(torch.load('small_llama_chat_model.pth'))
model.eval()
print('Model loaded from small_llama_chat_model.pth')

# Generate chat responses using the fine-tuned model
prompt = 'The quick brown fox'
max_length = 100
generated_text = generate_text(model, tokenizer, prompt, max_length)
print(generated_text)