#!/usr/bin/env python3
"""
Auto-Router-Ingestion: parst ein PDF mit Docling, klassifiziert jede Seite und
schickt sie automatisch zur passenden Methode:

  - Tabellen-Region        -> LLM-Verbalisierung
  - normaler Text          -> LLM-Verbalisierung
  - Fragment-Seite         -> Vision (Sicherheitsnetz fuer Infografiken, die
                              Docling faelschlich als viele kleine Text-Items labelt)

Ergebnis -> knowledge.jsonl. Laeuft in der Ingest-venv (docling + pymupdf + requests).

    python auto_ingest.py pdfs/mein-spiel.pdf
"""
import sys, os, json, base64
from collections import defaultdict
import requests
from docling.document_converter import DocumentConverter
import pymupdf

BASE   = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM    = os.environ.get("LLM_MODEL", "qwen3:14b")
VMODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
KNOW   = os.path.join(BASE, "knowledge.jsonl")
DPI    = int(os.environ.get("DPI", 150))
# Seiten mit auffaellig vielen Text-Fragmenten sind wahrscheinlich Infografiken,
# die Docling als Text labelt -> ueber Vision behandeln.
FRAGMENT_THRESHOLD = int(os.environ.get("FRAGMENT_THRESHOLD", 50))

VERB_PROMPT = ("Formuliere den folgenden Ausschnitt aus einem Brettspiel-Regelheft in "
               "vollstaendige, eigenstaendige deutsche Saetze um. Uebernimm ALLE Zahlen, "
               "Namen und Werte EXAKT. Erfinde nichts, lass nichts weg. Bei Tabellen: pro "
               "Zeile ein Satz mit Bezeichnung UND Wert. Antworte nur mit dem Text.")

VISION_PROMPT = ("Auf dem Bild ist ein Diagramm von Mitarbeiterkarten eines Brettspiels. "
                 "Liste JEDE Karte einzeln auf, ein Eintrag pro Zeile im Format "
                 "'Kartenname: Effekt und alle Werte'. Uebernimm alle Zahlen (Reichweite, "
                 "maximale Dauer, Kosten) exakt vom Bild. Erfinde nichts. Antworte auf Deutsch.")


def verbalize(text):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM,
        "messages": [{"role": "system", "content": VERB_PROMPT},
                     {"role": "user", "content": text}],
        "think": False, "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def vision(pdf, pageno):
    pix = pymupdf.open(pdf)[pageno - 1].get_pixmap(dpi=DPI)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": VMODEL, "prompt": VISION_PROMPT, "images": [b64], "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else next(
        iter(sorted(__import__("glob").glob(os.path.join(BASE, "pdfs", "*.pdf")))), None)
    if not pdf:
        sys.exit("Kein PDF. Nutzung: python auto_ingest.py <pdf>")

    print(f"Docling parst {pdf} ...")
    doc = DocumentConverter().convert(pdf).document

    pages = defaultdict(lambda: {"texts": [], "tables": [], "count": 0})
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        p = prov[0].page_no
        if type(item).__name__ == "TableItem":
            try:
                pages[p]["tables"].append(item.export_to_markdown())
            except Exception:
                pass
        else:
            t = getattr(item, "text", "") or ""
            if t:
                pages[p]["texts"].append(t)
            pages[p]["count"] += 1

    chunks, cid = [], 0
    for p in sorted(pages):
        info = pages[p]
        if info["count"] > FRAGMENT_THRESHOLD:
            route = "vision"
            out = vision(pdf, p)
            for ln in out.splitlines():
                ln = ln.strip().lstrip("-*0123456789. ").strip()
                if ":" in ln and len(ln) > 5:
                    cid += 1
                    chunks.append({"id": cid, "seite": p, "route": route, "text": ln})
            # Gemischte Seite: von Docling erkannte Tabellen ZUSAETZLICH verbalisieren.
            # Der Vision-Prompt ist auf Karten getrimmt und uebersieht sie sonst.
            for tbl in info["tables"]:
                cid += 1
                chunks.append({"id": cid, "seite": p, "route": "table",
                               "text": verbalize("Tabelle:\n" + tbl)})
        else:
            route = "text/table"
            for tbl in info["tables"]:
                cid += 1
                chunks.append({"id": cid, "seite": p, "route": "table",
                               "text": verbalize("Tabelle:\n" + tbl)})
            # Reiner Fliesstext wird ROH uebernommen: Docling liefert bereits saubere
            # Saetze, und Verbalisierung wuerde hier nur Werte verwaessern
            # (aus "50 % Bonus" wird "ein bestimmter Prozentsatz"). Nur Tabellen und
            # Grafiken brauchen die Umwandlung.
            txt = "\n".join(info["texts"]).strip()
            if txt:
                cid += 1
                chunks.append({"id": cid, "seite": p, "route": "text-roh", "text": txt})
        print(f"  Seite {p:>2}: route={route:11s} (text-items={info['count']}, tabellen={len(info['tables'])})")

    with open(KNOW, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} Chunks -> {KNOW}")
    routes = defaultdict(int)
    for c in chunks:
        routes[c["route"]] += 1
    print("Routen:", dict(routes))


if __name__ == "__main__":
    main()
