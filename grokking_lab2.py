# Setup and Imports
import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
from transformer_lens import HookedTransformer, HookedTransformerConfig
import transformer_lens.utils as utils
import tqdm.auto as tqdm
import copy

# Use the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 1. Model and Task Configuration ---
p = 113  # Modulus for our arithmetic task
frac_train = 0.3 # Fraction of the data to use for training

# Model Configuration
cfg = HookedTransformerConfig(
    n_layers=1,
    n_heads=4,
    d_model=128,
    d_head=32,
    d_mlp=512,
    act_fn="relu",
    normalization_type=None,
    d_vocab=p + 1,
    d_vocab_out=p,
    n_ctx=3,
    init_weights=True,
    device=device,
    seed=999,
)

model = HookedTransformer(cfg)

# Disable biases for simplicity
for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False


# --- 2. Dataset Generation ---
a_vector = torch.arange(p).repeat(p)
b_vector = torch.arange(p).repeat_interleave(p)
equals_vector = torch.full_like(a_vector, p)

dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
labels = (dataset[:, 0] + dataset[:, 1]) % p

# Train/Test Split
torch.manual_seed(42)
indices = torch.randperm(p * p)
cutoff = int(p * p * frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]

train_data = dataset[train_indices]
train_labels = labels[train_indices]
test_data = dataset[test_indices]
test_labels = labels[test_indices]


# --- 3. Training ---
def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
    return -correct_log_probs.mean()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

num_epochs = 25000
train_losses, test_losses = [], []

for epoch in tqdm.tqdm(range(num_epochs)):
    train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        test_logits = model(test_data)
        test_loss = loss_fn(test_logits, test_labels)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss {train_loss.item():.4f}, Test Loss {test_loss.item():.4f}")


# --- 4. Visualization ---
def plot_loss_curves(train_losses, test_losses):
    fig = px.line(y=[train_losses, test_losses], log_y=True,
                  labels={"x": "Epoch", "y": "Loss", "variable": "Dataset"},
                  title="Training and Test Loss for Modular Addition")
    fig.data[0].name = "Train"
    fig.data[1].name = "Test"
    fig.show()

plot_loss_curves(train_losses, test_losses)