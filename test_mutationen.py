#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutationsprobe: verfaelscht die Implementierung gezielt und prueft, dass ein
Test rot wird. Ein gruener Testlauf beweist nichts, solange nicht gezeigt ist,
dass die Tests ueberhaupt etwas festhalten.

    python test_mutationen.py

Jede Mutation wird eingespielt, test_classify.py und test_ingest_seite.py
laufen, dann wird die Datei aus dem Speicher zurueckgeschrieben (finally).
Exit-Code 1, wenn eine Mutation gruen bleibt, deren Namen nicht mit "[gleich]"
beginnt -- solche sind nachweislich verhaltensgleich, siehe M3.
"""
import io
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TESTS = ("test_classify.py", "test_ingest_seite.py")

# (Name, Datei, Suchmuster, Ersatz)
MUTATIONEN = [
    ("M1 Negationserkennung aus (NEG_FENSTER = 0)",
     "classify.py", "NEG_FENSTER = 2", "NEG_FENSTER = 0"),

    ("M2 Reihenfolge zurueckgedreht: Substring, flavor/meta zuerst",
     "classify.py",
     "    treffer = _nennungen(text)\n"
     "    kandidaten = {kat for kat, exakt in treffer if exakt}\n"
     "    genannt = {kat for kat, _ in treffer}\n"
     "    if len(kandidaten) == 1 and len(genannt) == 1:\n"
     "        return next(iter(kandidaten))\n"
     "    return None",
     '    for kat in ("flavor", "meta", "regel"):\n'
     "        if kat in text:\n"
     "            return kat\n"
     "    return None"),

    # Schritt 1 ist ein Schnellpfad: der exakte Vergleich in Schritt 2 deckt
    # ihn semantisch ab (0 Abweichungen ueber 322 Eingaben). Bleibt gruen.
    ("[gleich] M3 exaktes Matching (Schritt 1) entfernt",
     "classify.py",
     "    a = antwort.strip().lower()\n    if a in KATEGORIEN:\n        return a",
     "    a = antwort.strip().lower()"),

    ("M4 <think>-Block wird nicht entfernt",
     "classify.py",
     '    rest = re.sub(r"<think>.*?(?:</think>|\\Z)", " ", a, flags=re.DOTALL)',
     "    rest = a"),

    ("M5 Fallback ist flavor statt regel",
     "classify.py", 'FALLBACK   = "regel"', 'FALLBACK   = "flavor"'),

    ("M6 Wortgrenze aufgeweicht: Kompositum zaehlt als Kandidat",
     "classify.py", "            treffer.append((kat, wort == kat))",
     "            treffer.append((kat, True))"),

    ("M7 ingest.py laesst 'seite' wieder weg",
     "ingest.py", '    return {"id": cid, "seite": seite, "roh": roh, "text": text}',
     '    return {"id": cid, "roh": roh, "text": text}'),

    ("M8 ingest.py schreibt die Chunk-ID als Seite",
     "ingest.py", '    return {"id": cid, "seite": seite, "roh": roh, "text": text}',
     '    return {"id": cid, "seite": cid, "roh": roh, "text": text}'),

    ("M9 ingest.py bildet Bloecke wieder ueber das Gesamtdokument",
     "ingest.py",
     "    out = []\n"
     "    for p in sorted(seiten):\n"
     '        for b in split_blocks("\\n\\n".join(seiten[p]), target):\n'
     "            out.append((p, b))\n"
     "    return out",
     '    alles = "\\n\\n".join("\\n\\n".join(seiten[p]) for p in sorted(seiten))\n'
     "    return [(1, b) for b in split_blocks(alles, target)]"),

    ("M10 vision_ingest.py laesst 'seite' wieder weg",
     "vision_ingest.py",
     '    return {"id": cid, "seite": seite,\n'
     '            "quelle": f"vision:{os.path.basename(img)}", "text": text}',
     '    return {"id": cid,\n'
     '            "quelle": f"vision:{os.path.basename(img)}", "text": text}'),

    ("M11 vision_ingest.py erfindet Seite 1, statt abzubrechen",
     "vision_ingest.py",
     "    s = seite_aus_bildname(pfad)\n    if s is None:\n        raise SystemExit(",
     "    s = seite_aus_bildname(pfad)\n    if s is None:\n        return 1\n"
     "    if False:\n        raise SystemExit("),

    ("M12 classify.py schreibt die Rohantwort nicht mit",
     "classify.py", '        e["typ"], e["typ_antwort"] = classify(e["text"])',
     '        e["typ"] = classify(e["text"])[0]'),

    ("M13 classify.py verliert beim Neuschreiben das 'seite'-Feld",
     "classify.py",
     '    with open(KNOW, "w") as f:\n        for e in entries:\n'
     '            f.write(json.dumps(e, ensure_ascii=False) + "\\n")',
     '    with open(KNOW, "w") as f:\n        for e in entries:\n'
     '            mager = {k: v for k, v in e.items() if k != "seite"}\n'
     '            f.write(json.dumps(mager, ensure_ascii=False) + "\\n")'),

    ("M14 Kontrollausgabe nennt die Modellantwort nicht",
     "classify.py",
     "            print(f\"  [{e['typ']}] Modell sagte: {e['typ_antwort'][:70]!r}\")",
     "            print(f\"  [{e['typ']}]\")"),
]


def rote_tests():
    rot = []
    for datei in TESTS:
        p = subprocess.run([sys.executable, os.path.join(BASE, datei)],
                           cwd=BASE, capture_output=True, text=True)
        if p.returncode == 0:
            continue
        namen = [z.split(" ")[1] for z in p.stderr.splitlines()
                 if z.startswith(("FAIL:", "ERROR:"))]
        rot += [f"{datei}::{n}" for n in namen] or [f"{datei}::(Abbruch)"]
    return rot


def main():
    unerwartet_gruen = []
    for name, datei, alt, neu in MUTATIONEN:
        pfad = os.path.join(BASE, datei)
        orig = io.open(pfad, encoding="utf-8").read()
        if alt not in orig:
            print(f"[FEHLT] {name}: Muster nicht mehr in {datei} -- Mutation nachziehen")
            unerwartet_gruen.append(name)
            continue
        try:
            io.open(pfad, "w", encoding="utf-8").write(orig.replace(alt, neu, 1))
            rot = rote_tests()
        finally:
            io.open(pfad, "w", encoding="utf-8").write(orig)
        print(f"[{'ROT  ' if rot else 'GRUEN'}] {name}")
        for r in rot:
            print(f"         {r}")
        if not rot and not name.startswith("[gleich]"):
            unerwartet_gruen.append(name)

    print()
    if unerwartet_gruen:
        print(f"{len(unerwartet_gruen)} Mutation(en) ohne roten Test -- da fehlt ein Test:")
        for n in unerwartet_gruen:
            print(f"  {n}")
        return 1
    print(f"{len(MUTATIONEN)} Mutationen geprueft, alle erwartungsgemaess.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
