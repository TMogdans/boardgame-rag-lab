#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutationsprobe: verfaelscht die Implementierung gezielt und prueft, dass ein
Test rot wird. Ein gruener Testlauf beweist nichts, solange nicht gezeigt ist,
dass die Tests ueberhaupt etwas festhalten.

    python test_mutationen.py

Jede Mutation wird eingespielt, die Tests aus TESTS laufen, dann wird die Datei aus dem Speicher zurueckgeschrieben (finally).
Exit-Code 1, wenn eine Mutation gruen bleibt, deren Namen nicht mit "[gleich]"
beginnt -- solche sind nachweislich verhaltensgleich, siehe M3.
"""
import io
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
TESTS = ("test_classify.py", "test_ingest_seite.py", "test_wertung.py", "test_openwebui_pipe.py")

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

    ("M16 Wertung wieder von DROP_TYPES abhaengig machen",
     "rag.py",
     '    if frage.get("erwartet_verweigerung"):\n        return KAT_VERWEIGERUNG',
     '    import os as _os\n'
     '    if frage.get("typ") in {x for x in _os.environ.get("DROP_TYPES", "").split(",") if x}:\n'
     '        return KAT_VERWEIGERUNG\n'
     '    if frage.get("erwartet_verweigerung"):\n        return KAT_VERWEIGERUNG'),

    ("M17 stiller seite-Fallback zurueck (Chunk-ID als Seitenzahl)",
     "rag.py", 'def chunk_seite(c):', 'def chunk_seite(c):\n    return c.get("seite", c["id"])'),

    ("M18 Chunk-Guard entfernt (Endlosschleife bei CHUNK_SIZE <= OVERLAP)",
     "rag.py", "def pruefe_chunk_konfiguration(size=None, overlap=None):",
     "def pruefe_chunk_konfiguration(size=None, overlap=None):\n    return\n\n\ndef _abgeschaltet(size=None, overlap=None):"),

    ("M15 Flexionsendung bei Phrasen entfernt (Falsch-Negativ 'keine Angaben')",
     "rag.py",
     '    if rechts and len(kw.split()) > 1 and kw[-1:].isalpha():\n'
     '        rechts = r"\\w{0,3}(?!\\w)"',
     "    pass"),

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

    # ---- Open-WebUI-Pipe (test_openwebui_pipe.py) ----
    ("P1 Pipe baut den Prompt selbst statt ueber rag.baue_nachrichten",
     "openwebui_pipe.py", "        return rag.baue_nachrichten(frage, hits), hits",
     '        return [{"role": "system", "content": rag.SYSTEM_PROMPT}, {"role": "user", "content": frage}], hits'),
    ("P2 CLI baut den Prompt anders als die Pipe",
     "rag.py", '"messages": baue_nachrichten(query, hits),', '"messages": [{"role": "user", "content": query}],'),
    ("P3 CHUNK_SIZE-Valve wirkt nicht (Container-Umgebung schlaegt durch)",
     "openwebui_pipe.py", "        rag.CHUNK_SIZE = v.CHUNK_SIZE\n", ""),
    ("P4 RERANK nicht fest aus",
     "openwebui_pipe.py", "        rag.RERANK = False  # braucht torch, das im Open-WebUI-Image fehlt\n", ""),
    ("P5 EMBED_MODEL-Valve wirkt nicht",
     "openwebui_pipe.py", "        rag.EMBED_MODEL = v.EMBED_MODEL\n", ""),
    ("P6 Defaults weichen von der gemessenen Konfiguration ab",
     "openwebui_pipe.py", 'CHUNK_SIZE: int = Field(400,', 'CHUNK_SIZE: int = Field(800,'),
    ("P7 DROP_TYPES nicht im Index-Key",
     "openwebui_pipe.py", "v.DROP_TYPES, v.EMBED_MODEL, v.OLLAMA_URL, v.CHUNK_SIZE", "v.EMBED_MODEL, v.OLLAMA_URL, v.CHUNK_SIZE"),
    ("P8 Cache-Key nur mtime",
     "openwebui_pipe.py", "return (st.st_mtime_ns, st.st_size, st.st_ino)", "return st.st_mtime_ns"),
    ("P9 rag.py wird nie neu geladen",
     "openwebui_pipe.py", "if key != self._rag_key:", "if self._rag is None:"),
    ("P10 Retrieval blockiert den Event-Loop (kein to_thread)",
     "openwebui_pipe.py", "await asyncio.to_thread(self._suche, frage, self.valves)", "self._suche(frage, self.valves)"),
    ("P11 retrieve ausserhalb des Locks (Modell-Globale koennen wechseln)",
     "openwebui_pipe.py", "            chunks, embs = self._lade_index(rag, v)\n            hits =",
     "            chunks, embs = self._lade_index(rag, v)\n        hits ="),
    ("P12 Fehler entkommen dem Chat",
     "openwebui_pipe.py", "        except Exception as e:\n            yield f", "        except ZeroDivisionError as e:\n            yield f"),
    ("P13 auch GeneratorExit wird als Fehler abgefangen",
     "openwebui_pipe.py", "        except Exception as e:\n            yield f", "        except BaseException as e:\n            yield f"),
    ("P14 Fusszeile geht in den Verlauf",
     "openwebui_pipe.py", '"content": ohne_fusszeile(text_von(m.get("content")))', '"content": text_von(m.get("content"))'),
    ("P15 Task-Anfragen laufen ins Retrieval",
     "openwebui_pipe.py", "        if task:", "        if task == 'title_generation':"),
    ("P16 Ollama-Fehlergrund geht verloren",
     "openwebui_pipe.py", 'raise RuntimeError(f"Ollama antwortet {r.status_code}: {grund}")', 'raise RuntimeError(f"Ollama antwortet {r.status_code}")'),
    ("P17 LLM-Valves wirken nicht",
     "openwebui_pipe.py", '"model": v.LLM_MODEL, "messages": nachrichten, "think": v.THINK', '"model": "qwen3:14b", "messages": nachrichten, "think": False'),
    # _lade_rag setzt beim Neuladen _index_key=None -> der Index wird ohnehin neu
    # gebaut; der rag-Key im Index-Key ist Absicherung, kein zusaetzliches Verhalten.
    ("[gleich] P18 rag-Key nicht im Index-Key",
     "openwebui_pipe.py", "key = (self._rag_key, v.KNOWLEDGE_PATH", "key = (v.KNOWLEDGE_PATH"),
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
