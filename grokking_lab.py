# grokking_lab.py
# Reproducible, well-documented grokking experiment on modular addition.
# Run:
#   python grokking_lab.py --device cuda --steps 100000 --weight_decay 1e-3 --train_fraction 0.5
#   (Try steps 50k–200k for a clear "grok"; 4090 recommended.)

import math, random, argparse, json, os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ------------------------------
# Data: modular addition
# ------------------------------

def generate_mod_add_dataset(P: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return all pairs (a, b) with a,b in [0..P-1], labeled as (a + b) mod P.
    """
    A, B = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    A = A.reshape(-1)
    B = B.reshape(-1)
    Y = (A + B) % P
    return A, B, Y

class ModAddDataset(Dataset):
    def __init__(self, A: np.ndarray, B: np.ndarray, Y: np.ndarray):
        self.A = torch.tensor(A, dtype=torch.long)
        self.B = torch.tensor(B, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self): return self.Y.shape[0]
    def __getitem__(self, idx): return self.A[idx], self.B[idx], self.Y[idx]

def split_train_test(A, B, Y, train_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    N = len(Y)
    idx = np.arange(N)
    rng.shuffle(idx)
    cut = int(train_fraction * N)
    tr, te = idx[:cut], idx[cut:]
    return (A[tr], B[tr], Y[tr]), (A[te], B[te], Y[te])

# ------------------------------
# Model: Embeddings + MLP head
# ------------------------------

class EmbedAddMLP(nn.Module):
    """
    Two embeddings (for a and b) -> sum -> ReLU MLP -> logits over P classes.
    Small, but expressive enough to learn algorithmic structure.
    """
    def __init__(self, P: int, d_model: int = 128, hidden: int = 256):
        super().__init__()
        self.P = P
        self.emb_a = nn.Embedding(P, d_model)
        self.emb_b = nn.Embedding(P, d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, P)

        # Xavier init helps stabilization
        nn.init.xavier_uniform_(self.emb_a.weight)
        nn.init.xavier_uniform_(self.emb_b.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        x = self.emb_a(a) + self.emb_b(b)   # [B, d_model]
        x = F.relu(self.fc1(x))
        return self.fc2(x)                  # [B, P]

    def weight_l2(self) -> float:
        with torch.no_grad():
            s = 0.0
            for p in self.parameters():
                s += (p.detach() ** 2).sum().item()
        return math.sqrt(s)

# ------------------------------
# Training
# ------------------------------

@dataclass
class Config:
    P: int = 97
    d_model: int = 128
    hidden: int = 256
    train_fraction: float = 0.5
    batch_size: int = 1024
    steps: int = 200_000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    eval_every: int = 200
    device: str = "cpu"
    seed: int = 123
    outdir: str = "runs/grokking_modadd"

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()

def evaluate(model, ds, device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loader = DataLoader(ds, batch_size=4096, shuffle=False)
        for a,b,y in loader:
            a=a.to(device); b=b.to(device); y=y.to(device)
            logits = model(a,b)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += loss.item()
            correct  += (logits.argmax(-1)==y).sum().item()
            total   += y.numel()
    model.train()
    return loss_sum/total, correct/total

def run(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.outdir)

    A, B, Y = generate_mod_add_dataset(cfg.P)
    (Atr,Btr,Ytr), (Ate,Bte,Yte) = split_train_test(A,B,Y, cfg.train_fraction, cfg.seed)
    train_ds = ModAddDataset(Atr,Btr,Ytr)
    test_ds  = ModAddDataset(Ate,Bte,Yte)

    model = EmbedAddMLP(cfg.P, cfg.d_model, cfg.hidden).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    log = {"step":[], "train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[], "weight_l2":[]}

    step = 0
    while step < cfg.steps:
        for a,b,y in train_loader:
            a=a.to(cfg.device); b=b.to(cfg.device); y=y.to(cfg.device)
            logits = model(a,b)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.eval_every == 0 or step == cfg.steps-1:
                tl, ta = evaluate(model, train_ds, cfg.device)
                vl, va = evaluate(model, test_ds, cfg.device)
                log["step"].append(step)
                log["train_loss"].append(tl);  log["train_acc"].append(ta)
                log["test_loss"].append(vl);   log["test_acc"].append(va)
                log["weight_l2"].append(model.weight_l2())
                # Optional: quick console heartbeat
                print(f"[{step:>7}] train_acc={ta:.3f} test_acc={va:.3f} ||θ||2={log['weight_l2'][-1]:.2f}")

            step += 1
            if step >= cfg.steps:
                break

    # Save logs + config
    with open(os.path.join(cfg.outdir, "log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f)

    # Plots (matplotlib, one chart per figure, no custom colors)
    def plot(xs, ys, title, xlabel, ylabel, fname):
        plt.figure()
        plt.plot(xs, ys)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(cfg.outdir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    xs = log["step"]
    plot(xs, log["train_acc"], "Train Accuracy", "Step", "Accuracy", "train_acc.png")
    plot(xs, log["test_acc"],  "Test Accuracy",  "Step", "Accuracy", "test_acc.png")
    plot(xs, log["train_loss"],"Train Loss",     "Step", "Loss",     "train_loss.png")
    plot(xs, log["test_loss"], "Test Loss",      "Step", "Loss",     "test_loss.png")
    plot(xs, log["weight_l2"], "Parameter L2 Norm", "Step", "||θ||₂", "weight_l2.png")

    print(f"Saved logs and plots to: {cfg.outdir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=int, default=97)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--train_fraction", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="runs/grokking_modadd")
    args = p.parse_args()

    cfg = Config(**vars(args))
    run(cfg)

if __name__ == "__main__":
    main()
