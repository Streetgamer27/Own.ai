from fastapi import FastAPI
from pydantic import BaseModel
import json, torch
from model import TinyTransformer

app = FastAPI(title="OwnAI")

ckpt = torch.load("model.pt", map_location="cpu")
meta = json.load(open("vocab.json", encoding="utf-8"))
stoi = meta["stoi"]
itos = {int(k):v for k,v in meta["itos"].items()}

model = TinyTransformer(ckpt["vocab_size"], block_size=ckpt["block_size"])
model.load_state_dict(ckpt["model"])
model.eval()

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8

@app.get("/")
def root():
    return {"name": "OwnAI", "status": "online"}

@app.post("/generate")
def generate(req: Request):
    ids = [stoi[c] for c in req.prompt if c in stoi]
    if not ids:
        return {"text": req.prompt, "warning": "Prompt contains no known characters."}
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, min(req.max_new_tokens, 300), req.temperature)[0].tolist()
    return {"text": "".join(itos[i] for i in out)}
