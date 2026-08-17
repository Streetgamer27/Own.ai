# OwnAI 0.1

A small language model built from scratch with PyTorch. It does not call ChatGPT,
Gemini, or another hosted language model.

## Run locally/server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python chat.py
```

The first version is intentionally small. Put training text in `data/train.txt`.
The model learns by next-token prediction.

## Continuous learning

`learn.py` takes approved text from `data/new_data.txt`, appends it to the
training corpus, and retrains from scratch. This is deliberately controlled:
the program does NOT automatically trust arbitrary web pages or its own output.

## Server API

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then POST JSON to `/generate`:

```json
{"prompt":"Hello","max_new_tokens":80}
```

This is a learning project, not a production-scale LLM. A CPU VPS can train
very small models, but a GPU server will be much faster.
