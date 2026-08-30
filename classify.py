#!/usr/bin/env python3
"""
Taggt jeden Chunk in knowledge.jsonl per LLM als regel/flavor/meta.

So laesst sich Flavor-/Meta-Rauschen beim Retrieval ausfiltern
(rag.py: DROP_TYPES="flavor,meta"). Im Zweifel wird "regel" vergeben --
lieber eine Regel behalten als faelschlich verwerfen.
"""
import os, json, requests

BASE   = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM    = os.environ.get("LLM_MODEL", "qwen3:14b")
KNOW   = os.path.join(BASE, "knowledge.jsonl")

PROMPT = """Klassifiziere den folgenden Ausschnitt aus einem Brettspiel-Regelheft in GENAU EINE Kategorie:
- regel: Spielregel, Mechanik, Karteneffekt, Wert, Ablauf, Aufbau.
- flavor: Werbetext, Erzaehlung, woertliche Rede, thematische Ausschmueckung ohne Regelinhalt.
- meta: Inhaltsverzeichnis, Impressum, Credits, Danksagung, reine Seitenzahlen.
Antworte NUR mit einem Wort: regel, flavor oder meta."""


def classify(text):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": text}],
        "think": False, "stream": False,
    }, timeout=300)
    r.raise_for_status()
    ans = r.json()["message"]["content"].strip().lower()
    for t in ("flavor", "meta", "regel"):   # flavor/meta zuerst pruefen
        if t in ans:
            return t
    return "regel"


entries = [json.loads(l) for l in open(KNOW)]
counts = {}
for e in entries:
    e["typ"] = classify(e["text"])
    counts[e["typ"]] = counts.get(e["typ"], 0) + 1

with open(KNOW, "w") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print("Klassifiziert:", counts)
print("--- als flavor/meta markiert (zur Kontrolle) ---")
for e in entries:
    if e["typ"] != "regel":
        print(f"  [{e['typ']}] {e['text'][:90]}")
