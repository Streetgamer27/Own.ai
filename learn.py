from pathlib import Path
import subprocess

new = Path("data/new_data.txt")
train = Path("data/train.txt")

text = new.read_text(encoding="utf-8").strip()
if not text:
    raise SystemExit("data/new_data.txt is empty.")

# Controlled learning: only explicitly supplied data is accepted.
with train.open("a", encoding="utf-8") as f:
    f.write("\n" + text + "\n")

new.write_text("", encoding="utf-8")
print("Approved data added. Retraining candidate model...")
subprocess.run(["python", "train.py"], check=True)
