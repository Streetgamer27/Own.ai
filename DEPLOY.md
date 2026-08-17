# VPS deployment

On an Ubuntu VPS:

```bash
sudo apt update
sudo apt install -y python3 python3-venv
git clone YOUR_REPO_URL ownai
cd ownai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn server:app --host 0.0.0.0 --port 8000
```

For a real public deployment, put Nginx/Caddy in front of the API and enable HTTPS.
Do not expose an unprotected training endpoint to the public internet.
