#!/usr/bin/env python3
"""
Vision-Ingestion: verbalisiert eine Grafik-/Infografik-Seite mit einem lokalen
Vision-Modell (via Ollama) und fuegt JEDE Karte als EIGENEN Chunk in knowledge.jsonl.

Einzel-Chunks statt eines Riesenblocks: So matcht "Campaign Manager ... Dauer 3"
praezise mit einer Dauer-Frage, statt in einer 30-Karten-Liste unterzugehen.
Idempotent -- ersetzt vorhandene vision:-Chunks.

Jeder Chunk traegt die physische PDF-Seite im Feld 'seite' (1-basiert, Deckblatt
= 1) -- dieselbe Konvention wie auto_ingest.py und das Golden Set. Sie steckt im
Dateinamen, den render.py vergibt (seite_6.png -> Seite 6; render.py rechnet die
0-Basiertheit von pymupdf selbst heraus). Ist der Name anders, muss die Seite
ausdruecklich mitgegeben werden -- geraten wird sie nicht.

    python vision_ingest.py seite_6.png
    python vision_ingest.py karten.png 6      # Seite ausdruecklich
    SEITE=6 python vision_ingest.py karten.png
"""
import sys, os, re, json, base64, requests

BASE   = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VMODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
KNOW   = os.path.join(BASE, "knowledge.jsonl")

PROMPT = ("Auf dem Bild ist ein Diagramm von Mitarbeiterkarten eines Brettspiels. "
          "Liste JEDE Karte einzeln auf, ein Eintrag pro Zeile im Format "
          "'Kartenname: Effekt und alle Werte'. Uebernimm alle Zahlen (Reichweite, "
          "maximale Dauer, Kosten) exakt vom Bild. Erfinde nichts. Antworte auf Deutsch.")


def seite_aus_bildname(pfad):
    """Physische Seite aus dem von render.py vergebenen Namen, sonst None."""
    m = re.search(r"seite[_-]?(\d+)", os.path.basename(pfad), re.IGNORECASE)
    return int(m.group(1)) if m else None


def bestimme_seite(pfad, argv_seite=None, env_seite=None):
    """Physische Seite -- ausdrueckliche Angabe schlaegt den Dateinamen.

    Ohne beides ein harter Fehler: eine erfundene Seitenzahl ist genau der
    Defekt ("Fundstelle: Seite 75"), der hier behoben wird.
    """
    for wert in (argv_seite, env_seite):
        if wert not in (None, ""):
            s = int(wert)
            if s < 1:
                raise SystemExit(f"Seite muss >= 1 sein (physische PDF-Seite), nicht {s}.")
            return s
    s = seite_aus_bildname(pfad)
    if s is None:
        raise SystemExit(
            f"Seitenzahl aus '{os.path.basename(pfad)}' nicht ableitbar. "
            "Entweder von render.py rendern lassen (seite_6.png) oder die "
            "physische PDF-Seite mitgeben: python vision_ingest.py <bild> <seite>")
    return s


def frag_vision(img):
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": VMODEL, "prompt": PROMPT, "images": [b64], "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()


def split_karten(text):
    """Vision-Ausgabe in Einzel-Karten: Zeilen, die "Name: Wert" enthalten."""
    karten = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("-*0123456789. ").strip()
        if ":" in ln and len(ln) > 5:
            karten.append(ln)
    return karten


def lade_ohne_vision(pfad):
    """Bestehende Wissensbasis ohne die alten vision:-Chunks (Idempotenz)."""
    entries = []
    if os.path.exists(pfad):
        with open(pfad) as f:
            for line in f:
                e = json.loads(line)
                if not str(e.get("quelle", "")).startswith("vision:"):
                    entries.append(e)
    return entries


def eintrag(cid, seite, img, text):
    return {"id": cid, "seite": seite,
            "quelle": f"vision:{os.path.basename(img)}", "text": text}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("Nutzung: python vision_ingest.py <bild.png> [seite]")
    img = argv[0]
    seite = bestimme_seite(img, argv[1] if len(argv) > 1 else None,
                           os.environ.get("SEITE"))

    karten = split_karten(frag_vision(img))

    entries = lade_ohne_vision(KNOW)
    maxid = max([e.get("id", 0) for e in entries], default=0)
    for c in karten:
        maxid += 1
        entries.append(eintrag(maxid, seite, img, c))

    with open(KNOW, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"{len(karten)} Vision-Karten-Chunks eingefuegt "
          f"(ein Chunk pro Karte, Seite {seite}).")
    for c in karten[:3]:
        print("  z.B.:", c)


if __name__ == "__main__":
    main()
