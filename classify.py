#!/usr/bin/env python3
"""
Taggt jeden Chunk in knowledge.jsonl per LLM als regel/flavor/meta.

So laesst sich Flavor-/Meta-Rauschen beim Retrieval ausfiltern
(rag.py: DROP_TYPES="flavor,meta"). Im Zweifel wird "regel" vergeben --
lieber eine Regel behalten als faelschlich verwerfen.

Die Modellantwort wird von auswerten() geprueft, nicht per Substring-Suche.
Substring-Suche hat die Zusage der Zeile darueber ins Gegenteil verkehrt:
"Das ist eine regel, kein flavor." wurde zu flavor, und bei
DROP_TYPES=flavor,meta fiel die Regel damit aus dem Index.
"""
import os, re, json, requests

BASE   = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM    = os.environ.get("LLM_MODEL", "qwen3:14b")
KNOW   = os.path.join(BASE, "knowledge.jsonl")

KATEGORIEN = ("regel", "flavor", "meta")
FALLBACK   = "regel"

# Negiert das Modell eine Kategorie ("kein flavor", "nicht um Flavor"), zaehlt sie
# nicht als Kandidat. Fenster: die zwei Woerter davor -- deckt "nicht nur flavor"
# und "nicht um Flavor" ab, ohne ein "nicht" am Satzanfang auf alles zu beziehen.
NEGATIONEN = frozenset((
    "kein", "keine", "keinen", "keinem", "keiner", "keins",
    "nicht", "weder", "noch", "statt", "anstatt", "ohne",
))
NEG_FENSTER = 2

PROMPT = """Klassifiziere den folgenden Ausschnitt aus einem Brettspiel-Regelheft in GENAU EINE Kategorie:
- regel: Spielregel, Mechanik, Karteneffekt, Wert, Ablauf, Aufbau.
- flavor: Werbetext, Erzaehlung, woertliche Rede, thematische Ausschmueckung ohne Regelinhalt.
- meta: Inhaltsverzeichnis, Impressum, Credits, Danksagung, reine Seitenzahlen.
Antworte NUR mit einem Wort: regel, flavor oder meta."""


def _nennungen(text):
    """(kategorie, exakt) je nicht-negierter Nennung in text (klein geschrieben).

    exakt=True heisst: die Kategorie ist ein eigenes Wort ("flavor-Text"),
    exakt=False heisst: sie steckt in einem Kompositum ("Spielregel").
    """
    woerter = [(m.group(0), m.start()) for m in re.finditer(r"[a-zäöüß]+", text)]
    treffer = []
    for i, (wort, _) in enumerate(woerter):
        for kat in KATEGORIEN:
            if kat not in wort:
                continue
            davor = {w for w, _ in woerter[max(0, i - NEG_FENSTER):i]}
            if davor & NEGATIONEN:
                continue
            treffer.append((kat, wort == kat))
    return treffer


def _entscheide(text):
    """Genau eine nicht-negierte Kategorie als eigenes Wort -> die. Sonst None.

    Mehrere genannte Kategorien sind ein Zweifelsfall und werden bewusst NICHT
    entschieden -- der Aufrufer faellt dann auf FALLBACK zurueck.
    """
    treffer = _nennungen(text)
    kandidaten = {kat for kat, exakt in treffer if exakt}
    genannt = {kat for kat, _ in treffer}
    if len(kandidaten) == 1 and len(genannt) == 1:
        return next(iter(kandidaten))
    return None


def auswerten(antwort):
    """Modellantwort -> regel/flavor/meta. Im Zweifel FALLBACK ("regel").

    1. Getrimmte, kleingeschriebene Antwort exakt gegen die Kategorien
       (der Prompt verlangt EIN Wort, und darauf antwortet das Modell oft brav).
    2. Sonst <think>-Block entfernen und die letzte nichtleere Zeile pruefen --
       Reasoning-Modelle haengen die Antwort hinten an.
    3. Sonst auf Wortgrenzen (nicht Substrings) und ohne negierte Nennungen.
    4. Sonst FALLBACK: lieber eine Regel behalten als faelschlich verwerfen.
    """
    a = antwort.strip().lower()
    if a in KATEGORIEN:
        return a

    rest = re.sub(r"<think>.*?(?:</think>|\Z)", " ", a, flags=re.DOTALL)

    zeilen = [z.strip() for z in rest.splitlines() if z.strip()]
    if zeilen:
        letzte = zeilen[-1]
        if letzte in KATEGORIEN:
            return letzte
        kat = _entscheide(letzte)
        if kat:
            return kat

    kat = _entscheide(rest)
    if kat:
        return kat

    return FALLBACK


def classify(text):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM,
        "messages": [{"role": "system", "content": PROMPT},
                     {"role": "user", "content": text}],
        "think": False, "stream": False,
    }, timeout=300)
    r.raise_for_status()
    return auswerten(r.json()["message"]["content"])


def main():
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


if __name__ == "__main__":
    main()
