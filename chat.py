import json, torch
from model import TinyTransformer

ckpt = torch.load("model.pt", map_location="cpu")
meta = json.load(open("vocab.json", encoding="utf-8"))
stoi = meta["stoi"]
itos = {int(k):v for k,v in meta["itos"].items()}
model = TinyTransformer(ckpt["vocab_size"], block_size=ckpt["block_size"])
model.load_state_dict(ckpt["model"])
model.eval()

while True:
    prompt = input("\nYou: ")
    if prompt.lower() in {"exit", "quit"}:
        break
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        print("AI: I don't know those characters yet.")
        continue
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, 120, 0.8)[0].tolist()
    print("AI:", "".join(itos[i] for i in out))
