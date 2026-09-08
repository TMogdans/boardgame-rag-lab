#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests fuer das 'seite'-Feld in ingest.py und vision_ingest.py (Fix-Liste A3).

    python test_ingest_seite.py

Laeuft ohne PDF, ohne Ollama und ohne docling: das Docling-Dokument wird als
Attrappe konstruiert, die Modellaufrufe werden ersetzt. Geprueft wird die
WIRKUNG -- der Inhalt der geschriebenen knowledge.jsonl --, nicht der Quelltext.

Der Ausgangsbefund war "Fundstelle: Seite 75" bei einem 15-seitigen Regelheft:
rag.py fiel auf die Chunk-ID zurueck, weil beide Skripte kein 'seite' schrieben.
test_kein_seitenwert_ueber_der_seitenzahl faengt genau das.
"""
import contextlib
import io as _io
import json
import os
import sys
import tempfile
import types
import unittest

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# docling gehoert in die separate Ingest-venv und ist hier meist nicht da.
try:  # pragma: no cover
    import docling.document_converter  # noqa: F401
except ImportError:
    _docling = types.ModuleType("docling")
    _dc = types.ModuleType("docling.document_converter")
    _dc.DocumentConverter = object          # wird im Test ersetzt
    _docling.document_converter = _dc
    sys.modules["docling"] = _docling
    sys.modules["docling.document_converter"] = _dc

import ingest            # noqa: E402
import vision_ingest     # noqa: E402

SEITEN_IM_HEFT = 15      # wie das Regelheft aus der Fix-Liste


class Prov:
    def __init__(self, page_no):
        self.page_no = page_no


class TextItem:
    def __init__(self, text, seite):
        self.text = text
        self.prov = [Prov(seite)]


class TableItem:
    """Klassenname ist die Weiche in ingest.sammle_seiten()."""

    def __init__(self, markdown, seite):
        self.prov = [Prov(seite)]
        self._md = markdown

    def export_to_markdown(self):
        return self._md


class ItemOhneProvenance:
    """z.B. der Dokument-Wurzelknoten -- hat keine Seite."""

    def __init__(self, text="Regelheft"):
        self.text = text
        self.prov = []


class FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        return [(i, 0) for i in self._items]


def beispiel_doc():
    """15 Seiten Text, Seite 11 lang genug fuer mehrere Bloecke, plus Tabelle."""
    items = [ItemOhneProvenance()]
    for s in range(1, SEITEN_IM_HEFT + 1):
        items.append(TextItem(f"Text von Seite {s}.", s))
    # Seite 11 traegt viel Text -> mehrere Bloecke, IDs laufen ueber 15 hinaus
    for n in range(30):
        items.append(TextItem(f"Absatz {n} auf Seite 11. " + "x" * 200, 11))
    items.append(TableItem("| Karte | Dauer |\n| --- | --- |\n| Campaign Manager | 3 |", 14))
    return FakeDoc(items)


class FakeConverter:
    def __init__(self, doc):
        self._doc = doc

    def convert(self, pfad):
        return types.SimpleNamespace(document=self._doc)


def lauf_ingest(doc=None, target=None):
    """ingest.main() gegen eine Doc-Attrappe; liefert die geschriebenen Saetze."""
    doc = beispiel_doc() if doc is None else doc
    alt = (ingest.DocumentConverter, ingest.verbalize, ingest.OUT, sys.argv, ingest.TARGET)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "knowledge.jsonl")
        try:
            ingest.DocumentConverter = lambda: FakeConverter(doc)
            ingest.verbalize = lambda t: f"verbalisiert: {t[:40]}"
            ingest.OUT = out
            if target is not None:
                ingest.TARGET = target
            sys.argv = ["ingest.py", "regelheft.pdf"]
            with contextlib.redirect_stdout(_io.StringIO()):
                ingest.main()
            with open(out) as f:
                return [json.loads(l) for l in f]
        finally:
            (ingest.DocumentConverter, ingest.verbalize, ingest.OUT,
             sys.argv, ingest.TARGET) = alt


def lauf_vision(argv, vorhandene=None, antwort=None):
    """vision_ingest.main() ohne Vision-Modell; liefert die geschriebenen Saetze."""
    antwort = antwort or ("Campaign Manager: Reichweite 2, maximale Dauer 3\n"
                          "- Pizzabaecker: Kosten 4\n"
                          "2. CFO: 50 % Bonus auf Bargeld\n"
                          "kurz\n")
    alt = (vision_ingest.KNOW, vision_ingest.frag_vision, os.environ.get("SEITE"))
    with tempfile.TemporaryDirectory() as tmp:
        know = os.path.join(tmp, "knowledge.jsonl")
        if vorhandene:
            with open(know, "w") as f:
                for e in vorhandene:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        try:
            os.environ.pop("SEITE", None)
            vision_ingest.KNOW = know
            vision_ingest.frag_vision = lambda img: antwort
            with contextlib.redirect_stdout(_io.StringIO()):
                vision_ingest.main(argv)
            with open(know) as f:
                return [json.loads(l) for l in f]
        finally:
            vision_ingest.KNOW, vision_ingest.frag_vision = alt[0], alt[1]
            if alt[2] is None:
                os.environ.pop("SEITE", None)
            else:
                os.environ["SEITE"] = alt[2]


class TestIngestSeite(unittest.TestCase):
    def test_jeder_chunk_hat_ein_seite_feld(self):
        for e in lauf_ingest():
            self.assertIn("seite", e, e)

    def test_kein_seitenwert_ueber_der_seitenzahl(self):
        # Der Ausgangsbefund: Seitenwerte bis 79 bei 15 Seiten, weil die
        # Chunk-ID einsprang. Hier laufen die IDs bewusst weit ueber 15.
        saetze = lauf_ingest()
        self.assertGreater(max(e["id"] for e in saetze), SEITEN_IM_HEFT,
                           "Testdaten erzeugen zu wenige Chunks, um den Fall zu treffen")
        for e in saetze:
            self.assertTrue(1 <= e["seite"] <= SEITEN_IM_HEFT,
                            f"Seite {e['seite']} bei {SEITEN_IM_HEFT} Seiten (id={e['id']})")

    def test_seite_ist_nicht_die_chunk_id(self):
        saetze = lauf_ingest()
        abweichend = [e for e in saetze if e["seite"] != e["id"]]
        self.assertTrue(abweichend, "seite folgt stumpf der id -- das war der Defekt")

    def test_seite_stimmt_mit_dem_inhalt(self):
        for e in lauf_ingest():
            if "Text von Seite" in e["roh"] and "Absatz" not in e["roh"]:
                erwartet = int(e["roh"].split("Text von Seite ")[1].split(".")[0])
                self.assertEqual(e["seite"], erwartet)

    def test_tabellenseite_wird_mitgezaehlt(self):
        tabellen = [e for e in lauf_ingest() if "Campaign Manager" in e["roh"]]
        self.assertTrue(tabellen)
        for e in tabellen:
            self.assertEqual(e["seite"], 14)

    def test_bloecke_laufen_nicht_ueber_seitengrenzen(self):
        # Ein Block mit Text zweier Seiten haette keine eindeutige Seite.
        doc = FakeDoc([TextItem("Kurz auf Seite 1.", 1), TextItem("Kurz auf Seite 2.", 2)])
        saetze = lauf_ingest(doc, target=100000)
        self.assertEqual([e["seite"] for e in saetze], [1, 2])

    def test_items_ohne_provenance_erzeugen_keine_seite(self):
        # Kein Platzhalter, kein Rateversuch: das Item faellt raus.
        saetze = lauf_ingest(FakeDoc([ItemOhneProvenance("Wurzel"),
                                      TextItem("Echter Text.", 3)]))
        self.assertEqual([e["seite"] for e in saetze], [3])
        self.assertNotIn("Wurzel", saetze[0]["roh"])

    def test_seitenkonvention_ist_prov_page_no(self):
        # Golden Set: 'seiten' = physische pypdf-Seiten (Deckblatt = 1).
        # Docling liefert prov[0].page_no 1-basiert -> uebernehmen, kein +/-1.
        saetze = lauf_ingest(FakeDoc([TextItem("Deckblatt.", 1)]))
        self.assertEqual(saetze[0]["seite"], 1)

    def test_blocks_mit_seite_ist_deterministisch_sortiert(self):
        seiten = {11: ["b"], 2: ["a"], 14: ["c"]}
        self.assertEqual([s for s, _ in ingest.blocks_mit_seite(seiten)], [2, 11, 14])


class TestVisionSeite(unittest.TestCase):
    def test_jeder_karten_chunk_hat_die_seite(self):
        saetze = lauf_vision(["seite_6.png"])
        self.assertEqual(len(saetze), 3)
        for e in saetze:
            self.assertIn("seite", e)
            self.assertEqual(e["seite"], 6)
            self.assertEqual(e["quelle"], "vision:seite_6.png")

    def test_seite_ist_nicht_die_chunk_id(self):
        saetze = lauf_vision(["seite_6.png"])
        self.assertNotEqual([e["seite"] for e in saetze], [e["id"] for e in saetze])

    def test_seite_ausdruecklich_per_argument(self):
        for e in lauf_vision(["karten.png", "6"]):
            self.assertEqual(e["seite"], 6)

    def test_argument_schlaegt_dateinamen(self):
        for e in lauf_vision(["seite_6.png", "7"]):
            self.assertEqual(e["seite"], 7)

    def test_ohne_ableitbare_seite_harter_fehler(self):
        # Lieber abbrechen als eine Seitenzahl erfinden.
        with self.assertRaises(SystemExit):
            lauf_vision(["karten.png"])

    def test_render_py_namenskonvention(self):
        # render.py schreibt seite_<physische Seite>.png (pymupdf 0-basiert,
        # dort steht doc[s-1]) -- die Zahl im Namen ist also 1-basiert.
        self.assertEqual(vision_ingest.seite_aus_bildname("/x/seite_11.png"), 11)
        self.assertEqual(vision_ingest.seite_aus_bildname("Seite-2.PNG"), 2)
        self.assertIsNone(vision_ingest.seite_aus_bildname("karten.png"))

    def test_seite_null_oder_negativ_abgelehnt(self):
        with self.assertRaises(SystemExit):
            vision_ingest.bestimme_seite("karten.png", "0")

    def test_bestehende_chunks_behalten_ihre_seite(self):
        vorher = [{"id": 1, "seite": 2, "text": "Regeltext"},
                  {"id": 2, "seite": 6, "quelle": "vision:seite_6.png", "text": "alt"}]
        saetze = lauf_vision(["seite_6.png"], vorhandene=vorher)
        self.assertEqual(saetze[0], vorher[0])                  # unveraendert
        self.assertTrue(all("alt" != e["text"] for e in saetze[1:]))  # alte weg
        self.assertEqual([e["seite"] for e in saetze], [2, 6, 6, 6])
        self.assertEqual([e["id"] for e in saetze], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
