#!/usr/bin/env python3
"""
Vision-Ingestion: verbalisiert eine Grafik-/Infografik-Seite mit einem lokalen
Vision-Modell (via Ollama) und fuegt JEDE Karte als EIGENEN Chunk in knowledge.jsonl.

Einzel-Chunks statt eines Riesenblocks: So matcht "Campaign Manager ... Dauer 3"
praezise mit einer Dauer-Frage, statt in einer 30-Karten-Liste unterzugehen.
Idempotent -- ersetzt vorhandene vision:-Chunks.

    python vision_ingest.py seite_6.png
"""
import sys, os, json, base64, requests

BASE   = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VMODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
KNOW   = os.path.join(BASE, "knowledge.jsonl")

PROMPT = ("Auf dem Bild ist ein Diagramm von Mitarbeiterkarten eines Brettspiels. "
          "Liste JEDE Karte einzeln auf, ein Eintrag pro Zeile im Format "
          "'Kartenname: Effekt und alle Werte'. Uebernimm alle Zahlen (Reichweite, "
          "maximale Dauer, Kosten) exakt vom Bild. Erfinde nichts. Antworte auf Deutsch.")

img = sys.argv[1]
with open(img, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = requests.post(f"{OLLAMA}/api/generate", json={
    "model": VMODEL, "prompt": PROMPT, "images": [b64], "stream": False,
}, timeout=600)
r.raise_for_status()
text = r.json()["response"].strip()

# bestehende Wissensbasis laden, alte vision:-Chunks entfernen (Idempotenz)
entries = []
if os.path.exists(KNOW):
    for line in open(KNOW):
        e = json.loads(line)
        if not str(e.get("quelle", "")).startswith("vision:"):
            entries.append(e)

maxid = max([e.get("id", 0) for e in entries], default=0)

# Vision-Ausgabe in Einzel-Karten aufsplitten: Zeilen, die "Name: Wert" enthalten
cards = []
for ln in text.splitlines():
    ln = ln.strip().lstrip("-*0123456789. ").strip()
    if ":" in ln and len(ln) > 5:
        cards.append(ln)

for c in cards:
    maxid += 1
    entries.append({"id": maxid, "quelle": f"vision:{os.path.basename(img)}", "text": c})

with open(KNOW, "w") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"{len(cards)} Vision-Karten-Chunks eingefuegt (ein Chunk pro Karte).")
for c in cards[:3]:
    print("  z.B.:", c)
