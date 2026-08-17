import json, math, os
from pathlib import Path
import torch
from model import TinyTransformer

DATA = Path("data/train.txt")
CKPT = Path("model.pt")
META = Path("vocab.json")
BLOCK = 128
BATCH = 32
STEPS = 1000
LR = 3e-4

text = DATA.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

if len(data) < BLOCK + 2:
    raise SystemExit("Add more text to data/train.txt before training.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyTransformer(len(chars), block_size=BLOCK).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

def batch():
    ix = torch.randint(0, len(data)-BLOCK-1, (BATCH,))
    x = torch.stack([data[i:i+BLOCK] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(device)
    return x, y

model.train()
for step in range(STEPS):
    x, y = batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        print(f"step {step:4d} loss {loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "vocab_size": len(chars),
    "block_size": BLOCK,
}, CKPT)
META.write_text(json.dumps({"stoi": stoi, "itos": {str(k):v for k,v in itos.items()}}, ensure_ascii=False), encoding="utf-8")
print(f"Saved {CKPT} on {device}")
