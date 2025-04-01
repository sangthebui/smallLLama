import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import math

# Load the dataset from Hugging Face
dataset = load_dataset('mikasenghaas/wikitext-2', split='train[:50%]')
print("Dataset loaded successfully.")

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define a custom dataset class for the dataset
class WikiText2Dataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        return inputs['input_ids'].flatten()

# Create a custom dataset instance
seq_len = 128
custom_dataset = WikiText2Dataset(dataset, tokenizer, seq_len)
print("Custom dataset created successfully.")

# Create a data loader
batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# # Define the simplified LLama architecture
class MiniLLama(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers):
        super(MiniLLama, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([self._create_layer(hidden_size, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, vocab_size)

    def _create_layer(self, hidden_size, num_heads):
        return nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.1)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        for layer in self.layers:
            embeddings = layer(embeddings, embeddings)
        outputs = self.fc(embeddings)
        return outputs

# Set hyperparameters
vocab_size = tokenizer.vocab_size
hidden_size = 256
num_heads = 4
num_layers = 4

# Initialize the model, optimizer, and loss function
model = MiniLLama(vocab_size, hidden_size, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("Model initialized successfully.")
# Train the model
for epoch in range(10):
    for batch in data_loader:
        input_ids = batch
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
# Save the model
torch.save(model.state_dict(), 'small_llama_model.pth')
print('Model saved to small_llama_model.pth')


# Load the saved model
model.load_state_dict(torch.load('small_llama_model.pth'))
model.eval()
print('Model loaded from small_llama_model.pth')

# Define a function to generate text using the model
def generate_text(model, tokenizer, prompt, max_length):
    # turn on the model evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model(input_ids)
    output = torch.argmax(output, dim=-1)
    output = output[:, -1]
    generated_text = []
    for _ in range(max_length):
        input_ids = torch.cat((input_ids, output.unsqueeze(0)), dim=1)
        output = model(input_ids)
        output = torch.argmax(output, dim=-1)
        output = output[:, -1]
        generated_text.append(tokenizer.decode(output, skip_special_tokens=True))
    return ''.join(generated_text)


# Generate text using the model
prompt = 'The quick brown fox'
max_length = 128
generated_text = generate_text(model, tokenizer, prompt, max_length)
print(generated_text)