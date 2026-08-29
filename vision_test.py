#!/usr/bin/env python3
"""Schickt ein Bild + eine Frage an ein lokales Vision-Modell (via Ollama).

    python vision_test.py seite_6.png "Was steht auf der Campaign-Manager-Karte?"
"""
import sys, os, base64, requests

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL  = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")

img    = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "Beschreibe dieses Bild."

with open(img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post(f"{OLLAMA}/api/generate", json={
    "model": MODEL,
    "prompt": prompt,
    "images": [b64],
    "stream": False,
}, timeout=600)
r.raise_for_status()
print(r.json()["response"].strip())
