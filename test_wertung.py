#!/usr/bin/env python3
"""
Tests der Auswertungslogik von rag.py -- laeuft OHNE Ollama, ohne Modelle, ohne PDF.

    python test_wertung.py            # oder: python -m unittest test_wertung -v

Nur Standardbibliothek (unittest), keine neuen Abhaengigkeiten. Moeglich ist das,
weil die Wertung in rag.py aus reinen Funktionen besteht, die weder Netz noch
Umgebungsvariablen lesen: bewerte_retrieval, bewerte_frage, keyword_treffer,
fasse_zusammen, zerteile, chunk_seite, lies_drop_types, baue_knowledge_chunks.

Kern des Ganzen ist test_drop_types_veraendert_die_wertung_nicht: die Gegenmessung
gegen die dokumentierte Tabelle aus der Befundliste (A1).
"""
import contextlib, io, json, os, re, sys, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag

BASE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = json.load(open(os.path.join(BASE, "golden_set.example.json")))
FRAGEN = GOLDEN["fragen"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Ein FIXIERTES Retrieval-Ergebnis: Seiten unter top_k je Frage-ID. Es ist so
# gewaehlt, dass es die in der Befundliste dokumentierte Messung exakt
# reproduziert (5 Treffer; Fehlschlaege bei 5, 6, 7 und 10). Damit ist es kein
# frei erfundenes Beispiel, sondern die Datenbasis der Tabelle aus A1 --
# test_alte_logik_reproduziert_die_dokumentierte_tabelle weist das nach.
ABGERUFEN = {
    1:  [11, 3, 4, 5],     # erwartet [11]     -> Treffer
    2:  [14, 1, 2, 3],     # erwartet [11, 14] -> Treffer
    3:  [11, 12, 1, 2],    # erwartet [11]     -> Treffer
    4:  [6, 7, 8, 9],      # erwartet [6, 10]  -> Treffer
    5:  [1, 2, 3, 4],      # erwartet [6]      -> Fehlschlag
    6:  [1, 2, 3, 4],      # erwartet [11, 14] -> Fehlschlag
    7:  [1, 3, 4, 5],      # erwartet [2]      -> Verweigerungsfrage
    8:  [1, 2, 3, 4],      # erwartet []       -> Verweigerungsfrage
    9:  [5, 6, 7, 8],      # erwartet [5]      -> Treffer
    10: [1, 2, 3, 4],      # erwartet [5]      -> Fehlschlag
}

# Wortwoertlich die gemessene faktenfreie Antwort aus der Befundliste (A5).
# Sie enthaelt keinen einzigen der gefragten Fakten und traf mit den alten
# Keywords 4 von 10 Fragen.
FAKTENFREI = (
    "Dazu kann ich nichts Genaues sagen. Es haengt von der Spielsituation ab; "
    "in Phase 4 passiert vieles gleichzeitig, und es gibt 10 bis 16 Karten im "
    "Spiel. Genauer steht das nicht im Dokument."
)

# Die DROP_TYPES-Varianten aus der Tabelle in A1, unveraendert.
DROP_VARIANTEN = ["", "flavor", "falle", "flavor,falle", "fakt,falle,flavor,tabelle"]


def werte_lauf_aus(antwort_je_frage=None):
    """Kompletter Wertungsdurchlauf ueber das Beispiel-Golden-Set mit fixem Retrieval."""
    saetze = [
        rag.bewerte_frage(f, ABGERUFEN[f["id"]], (antwort_je_frage or {}).get(f["id"], ""))
        for f in FRAGEN
    ]
    return saetze, rag.fasse_zusammen(saetze)


def alte_wertung(drop_types):
    """Die Logik von vor der Reparatur, zum Vergleich nachgebaut.

    Genau die Zeile, die zurueckgenommen wurde:
        pruefbar = bool(erwartete) and f.get("typ") not in gefilterte_typen
    """
    drop = {x for x in drop_types.split(",") if x}
    treffer = nenner = 0
    for f in FRAGEN:
        erwartete = f.get("seiten") or []
        if not (bool(erwartete) and f.get("typ") not in drop):
            continue
        nenner += 1
        treffer += int(any(s in ABGERUFEN[f["id"]] for s in erwartete))
    return treffer, nenner


# ---------------------------------------------------------------------------
# A1 -- Dreiteilung getroffen / verfehlt / Verweigerungsfrage
# ---------------------------------------------------------------------------
class TestDreiteilung(unittest.TestCase):
    def test_getroffen(self):
        f = {"id": 1, "seiten": [11]}
        self.assertEqual(rag.bewerte_retrieval(f, [11, 3, 4]), rag.KAT_GETROFFEN)

    def test_verfehlt(self):
        f = {"id": 5, "seiten": [6]}
        self.assertEqual(rag.bewerte_retrieval(f, [1, 2, 3]), rag.KAT_VERFEHLT)

    def test_verweigerungsfrage_auch_mit_erwarteter_seite(self):
        """Frage 7 hat seiten [2] UND erwartet_verweigerung -- das Feld gewinnt."""
        f = next(x for x in FRAGEN if x["id"] == 7)
        self.assertEqual(f["seiten"], [2])
        self.assertTrue(f["erwartet_verweigerung"])
        self.assertEqual(rag.bewerte_retrieval(f, [1, 3, 4, 5]), rag.KAT_VERWEIGERUNG)
        self.assertEqual(rag.bewerte_retrieval(f, [2, 3, 4, 5]), rag.KAT_VERWEIGERUNG)

    def test_verweigerungsfrage_ohne_erwartete_seite(self):
        f = next(x for x in FRAGEN if x["id"] == 8)
        self.assertEqual(f["seiten"], [])
        self.assertEqual(rag.bewerte_retrieval(f, [1, 2, 3, 4]), rag.KAT_VERWEIGERUNG)

    def test_typ_ist_keine_wertungsanweisung(self):
        """Ein Frage-Typ allein darf nichts steuern -- nur das explizite Feld."""
        f = {"id": 99, "typ": "flavor", "seiten": [2]}          # typ flavor, kein Feld
        self.assertEqual(rag.bewerte_retrieval(f, [1, 2, 3]), rag.KAT_GETROFFEN)
        f2 = {"id": 98, "typ": "fakt", "seiten": [7], "erwartet_verweigerung": True}
        self.assertEqual(rag.bewerte_retrieval(f2, [7]), rag.KAT_VERWEIGERUNG)

    def test_frage_ohne_seiten_und_ohne_verweigerung_ist_fehler(self):
        """Keine stille Nennerkuerzung: so eine Frage ist ein Golden-Set-Defekt."""
        with self.assertRaises(rag.KonfigFehler) as ctx:
            rag.bewerte_retrieval({"id": 42, "typ": "fakt", "seiten": []}, [1, 2])
        self.assertIn("erwartet_verweigerung", str(ctx.exception))

    def test_nenner_der_dreiteilung_geht_auf(self):
        _, z = werte_lauf_aus()
        self.assertEqual(z["fragen"], 10)
        self.assertEqual(z["retrieval_nenner"], 8)
        self.assertEqual(z["verweigerungsfragen"], 2)
        self.assertEqual(z["getroffen"] + z["verfehlt"], z["retrieval_nenner"])
        self.assertEqual(z["retrieval_nenner"] + z["verweigerungsfragen"], z["fragen"])
        self.assertEqual((z["getroffen"], z["verfehlt"]), (5, 3))


# ---------------------------------------------------------------------------
# A1/A2 -- Gegenmessung: DROP_TYPES darf die Wertung nicht mehr beruehren
# ---------------------------------------------------------------------------
class TestDropTypesUndWertung(unittest.TestCase):
    def test_alte_logik_reproduziert_die_dokumentierte_tabelle(self):
        """Beweist, dass das Fixture die Messung aus A1 ist -- nicht ein neues Beispiel."""
        self.assertEqual(alte_wertung(""), (5, 9))                            # 56 %
        self.assertEqual(alte_wertung("flavor"), (5, 8))                      # 62 %
        self.assertEqual(alte_wertung("falle"), (5, 7))                       # 71 %
        self.assertEqual(alte_wertung("flavor,falle"), (5, 6))                # 83 %
        self.assertEqual(alte_wertung("fakt,falle,flavor,tabelle"), (0, 0))   # 0/0

    def test_drop_types_veraendert_die_wertung_nicht(self):
        """Identisches Retrieval-Ergebnis, alle DROP_TYPES-Varianten: eine Zahl."""
        referenz = None
        for variante in DROP_VARIANTEN:
            for quelle in ("knowledge", "pdf"):
                with self.subTest(drop_types=variante, source=quelle):
                    with mock.patch.dict(os.environ,
                                         {"DROP_TYPES": variante, "SOURCE": quelle}):
                        _, z = werte_lauf_aus()
                    if referenz is None:
                        referenz = z
                    self.assertEqual(z, referenz)
        self.assertEqual((referenz["getroffen"], referenz["retrieval_nenner"]), (5, 8))
        self.assertEqual(referenz["verweigerungsfragen"], 2)

    def test_drop_types_unbekannter_wert_ist_fehler(self):
        for wert in ("falle", "fakt,falle,flavor,tabelle", "flavour", "leerstelle"):
            with self.subTest(wert=wert):
                with self.assertRaises(rag.KonfigFehler) as ctx:
                    rag.lies_drop_types({"DROP_TYPES": wert, "SOURCE": "knowledge"})
                self.assertIn("classify.py", str(ctx.exception))

    def test_drop_types_ohne_knowledge_source_ist_fehler(self):
        for quelle in ({}, {"SOURCE": "pdf"}, {"SOURCE": ""}):
            with self.subTest(env=quelle):
                env = {"DROP_TYPES": "flavor"}
                env.update(quelle)
                with self.assertRaises(rag.KonfigFehler) as ctx:
                    rag.lies_drop_types(env)
                self.assertIn("SOURCE=knowledge", str(ctx.exception))

    def test_drop_types_gueltig_wird_akzeptiert(self):
        self.assertEqual(
            rag.lies_drop_types({"DROP_TYPES": "flavor,meta", "SOURCE": "knowledge"}),
            {"flavor", "meta"})
        self.assertEqual(rag.lies_drop_types({}), set())
        self.assertEqual(rag.lies_drop_types({"DROP_TYPES": ""}), set())

    def test_drop_types_ohne_typ_feld_ist_fehler(self):
        roh = [{"id": "k1", "seite": 3, "text": "Regeltext"},
               {"id": "k2", "seite": 4, "text": "Mehr Regeltext"}]
        with self.assertRaises(rag.KonfigFehler) as ctx:
            rag.baue_knowledge_chunks(roh, {"flavor"})
        self.assertIn("classify.py zuerst laufen lassen", str(ctx.exception))
        # ohne DROP_TYPES ist ein fehlendes typ-Feld voellig in Ordnung
        self.assertEqual(len(rag.baue_knowledge_chunks(roh, set())), 2)

    def test_drop_types_filtert_den_index_weiterhin(self):
        """Der Filter soll wirken -- nur eben auf den Index, nicht auf die Wertung."""
        roh = [{"id": "k1", "seite": 2, "text": "Werbespruch", "typ": "flavor"},
               {"id": "k2", "seite": 3, "text": "Regeltext", "typ": "regel"}]
        self.assertEqual(len(rag.baue_knowledge_chunks(roh, set())), 2)
        uebrig = rag.baue_knowledge_chunks(roh, {"flavor"})
        self.assertEqual([c["seite"] for c in uebrig], [3])


# ---------------------------------------------------------------------------
# A3 (rag.py-Seite) -- fehlendes seite-Feld ist ein Fehler, kein Fallback
# ---------------------------------------------------------------------------
class TestSeitenfeld(unittest.TestCase):
    def test_chunk_seite_ohne_feld_ist_fehler(self):
        with self.assertRaises(rag.KonfigFehler) as ctx:
            rag.chunk_seite({"id": "k79", "text": "irgendwas"})
        meldung = str(ctx.exception)
        self.assertIn("k79", meldung)
        for skript in ("auto_ingest.py", "ingest.py", "vision_ingest.py"):
            self.assertIn(skript, meldung)

    def test_chunk_seite_mit_feld(self):
        self.assertEqual(rag.chunk_seite({"id": "k1", "seite": 11, "text": "x"}), 11)

    def test_chunk_id_wandert_nicht_in_die_seite(self):
        """Der eigentliche Befund: aus Chunk 79 wurde 'Seite 79' bei 15 Seiten."""
        roh = [{"id": 79, "text": "Ein Chunk ohne Seitenangabe"}]
        with self.assertRaises(rag.KonfigFehler):
            rag.baue_knowledge_chunks(roh, set())

    def test_seite_wird_uebernommen(self):
        roh = [{"id": "k1", "seite": 11, "text": "A" * 10}]
        chunks = rag.baue_knowledge_chunks(roh, set())
        self.assertEqual([c["seite"] for c in chunks], [11])


# ---------------------------------------------------------------------------
# A5 -- Keywords: Wortgrenzen und kontextualisierte Stichwoerter
# ---------------------------------------------------------------------------
class TestKeywords(unittest.TestCase):
    def test_nicht_steckt_nicht_mehr_in_nichts(self):
        self.assertEqual(rag.keyword_treffer(["nicht"], "Dazu kann ich nichts sagen."), [])
        self.assertEqual(rag.keyword_treffer(["nicht"], "Das steht nicht im Dokument."), ["nicht"])

    def test_wortgrenze_allein_reicht_nicht(self):
        """Deshalb wurden die Einzelziffern zusaetzlich kontextualisiert.

        Eine nackte '10' ist auch mit Wortgrenzen von 'es gibt 10 bis 16 Karten'
        erfuellbar -- der Fix steckt zur Haelfte im Golden Set, nicht im Regex.
        """
        self.assertEqual(rag.keyword_treffer(["10"], "es gibt 10 bis 16 Karten"), ["10"])
        self.assertEqual(rag.keyword_treffer(["10 Dollar"], "es gibt 10 bis 16 Karten"), [])

    def test_sonderzeichen_keywords_funktionieren(self):
        self.assertEqual(rag.keyword_treffer(["50 %"], "Der CFO bringt 50 % Bonus."), ["50 %"])
        self.assertEqual(rag.keyword_treffer(["50 %"], "Der Bonus liegt bei 150 % davon."), [])
        self.assertEqual(rag.keyword_treffer(["#12"], "Es fliegen #12, #15 und #16 raus."), ["#12"])
        self.assertEqual(rag.keyword_treffer(["#12"], "Nummer #123 bleibt drin."), [])
        self.assertEqual(rag.keyword_treffer(["4x4"], "Der Plan ist 4x4 Felder gross."), ["4x4"])
        self.assertEqual(rag.keyword_treffer(["2-5"], "Fuer 2-5 Spieler."), ["2-5"])
        self.assertEqual(rag.keyword_treffer(["$10"], "Preis $10 pro Stueck."), ["$10"])

    def test_gross_klein_egal(self):
        self.assertEqual(rag.keyword_treffer(["Essenszeit"], "in der ESSENSZEIT"), ["Essenszeit"])

    def test_faktenfreie_antwort_trifft_kaum_noch_keywords(self):
        """Gemessene Gegenprobe: vorher 4 von 10 Fragen, davon 3 Falsch-Positive."""
        antworten = {f["id"]: FAKTENFREI for f in FRAGEN}
        saetze, z = werte_lauf_aus(antworten)
        getroffen = {s["id"]: s["keywords_getroffen"] for s in saetze if s["keywords_getroffen"]}
        # Die drei dokumentierten Falsch-Positive sind weg:
        # Frage 1 ("10" aus "10 bis 16"), Frage 8 ("nicht" aus "nichts"),
        # Frage 10 ("16" aus "10 bis 16").
        for fid in (1, 8, 10):
            self.assertNotIn(fid, getroffen, f"Frage {fid} trifft weiter faelschlich")
        # Uebrig bleibt genau Frage 3: die Antwort nennt woertlich "Phase 4" und
        # damit tatsaechlich den gemeinten Fakt -- kein Artefakt des Matchings.
        self.assertEqual(getroffen, {3: ["Phase 4"]})
        self.assertEqual(z["kw_treffer"], 1)
        self.assertEqual(z["kw_nenner"], 10)

    def test_keine_nackten_zahlen_im_golden_set(self):
        """Genau der Defekt aus A5: eine einzelne Zahl ohne Kontext, oder ein Einzelzeichen.

        "10", "3", "4", "16" und "%" waren von beliebigem Fliesstext erfuellbar.
        Erlaubt bleibt alles, was einen Kontextanker mittraegt -- ein Wort
        ("Phase 4"), ein Waehrungszeichen ("3$"), eine Aufzaehlung ("12, 15, 16").
        """
        for f in FRAGEN:
            for kw in f["keywords"]:
                kern = kw.strip()
                with self.subTest(frage=f["id"], keyword=kw):
                    self.assertGreater(len(kern), 1,
                                       f"Frage {f['id']}: Einzelzeichen {kw!r} als Keyword")
                    self.assertIsNone(re.fullmatch(r"\d+", kern),
                                      f"Frage {f['id']}: nackte Zahl {kw!r} als Keyword")

    def test_keyword_nicht_bei_frage_8_entfernt(self):
        f8 = next(x for x in FRAGEN if x["id"] == 8)
        self.assertNotIn("nicht", f8["keywords"])

    def test_richtige_antworten_treffen_weiter(self):
        """Gegenprobe zur Verscharfung: die Metrik darf nicht einfach nur strenger sein."""
        richtig = {
            1:  "Eine Ware kostet standardmaessig 10 Dollar (Seite 11).",
            2:  "Der CFO gibt 50 % Bonus auf das verdiente Bargeld (Seite 11).",
            3:  "Das Geld wird in Phase 4, der Essenszeit, verdient (Seite 11).",
            4:  "Der Truck Driver hat eine Reichweite von 3 (Seite 6).",
            5:  "Die Kampagne hat eine maximale Dauer von 3 (Seite 6).",
            6:  "Eine Waitress bringt 3 Dollar (Seite 11).",
            7:  "Nein, das ist Werbetext vom Deckblatt und keine Regel (Seite 2).",
            8:  "Dazu enthaelt das Dokument keine Angaben; genannt sind 2 bis 5 Spieler.",
            9:  "Bei 4 Spielern ist der Spielplan 4x4 Felder gross (Seite 5).",
            10: "Entfernt werden die Reklametafeln 12, 15 und 16 (Seite 5).",
        }
        saetze, z = werte_lauf_aus(richtig)
        leer = [s["id"] for s in saetze if not s["keywords_getroffen"]]
        self.assertEqual(leer, [], f"plausible richtige Antwort trifft kein Keyword: {leer}")
        self.assertEqual(z["kw_treffer"], 10)

    def test_verweigerungssignal_wird_getrennt_gezaehlt(self):
        antworten = {7: "Nein, das ist Werbetext.", 8: "Es gibt einen Solo-Modus mit 7 Karten."}
        _, z = werte_lauf_aus(antworten)
        self.assertEqual(z["verweigerungsfragen"], 2)
        self.assertEqual(z["verweigerung_signal"], 1)


# ---------------------------------------------------------------------------
# A5 -- Ausgabe: beide Nenner explizit, ehrliches Etikett
# ---------------------------------------------------------------------------
class TestAusgabe(unittest.TestCase):
    def test_zusammenfassung_nennt_beide_nenner(self):
        _, z = werte_lauf_aus()
        text = "\n".join(rag.formatiere_zusammenfassung(z))
        self.assertIn("getroffen : 5/8", text)
        self.assertIn("verfehlt  : 3/8", text)
        self.assertIn("Verweigerungsfragen: 2/10", text)
        self.assertIn("0/10", text)          # Keyword-Nenner, hier ohne Antworten
        self.assertIn("Fragen im Golden Set: 10", text)

    def test_zusammenfassung_nennt_die_metrik_ehrlich(self):
        _, z = werte_lauf_aus()
        text = "\n".join(rag.formatiere_zusammenfassung(z))
        self.assertIn("Regressionswarner", text)
        self.assertIn("KEIN Korrektheitsmass", text)
        self.assertIn("Indiz, kein Urteil", text)

    def test_keine_automatische_korrektheitsbewertung(self):
        """Bewusste Grenze: das Skript erhebt 'falsch' nicht und behauptet es nicht."""
        z = rag.fasse_zusammen(werte_lauf_aus()[0])
        self.assertNotIn("falsch", z)
        self.assertNotIn("korrekt", z)


# ---------------------------------------------------------------------------
# A6 -- Guard gegen die Endlosschleife
# ---------------------------------------------------------------------------
class TestChunkGuard(unittest.TestCase):
    def test_size_gleich_overlap_ist_fehler(self):
        with self.assertRaises(rag.KonfigFehler) as ctx:
            rag.pruefe_chunk_konfiguration(150, 150)
        self.assertIn("CHUNK_OVERLAP", str(ctx.exception))

    def test_size_kleiner_overlap_ist_fehler(self):
        with self.assertRaises(rag.KonfigFehler):
            rag.pruefe_chunk_konfiguration(100, 150)

    def test_zerteile_bricht_ab_statt_zu_haengen(self):
        with self.assertRaises(rag.KonfigFehler):
            rag.zerteile("A" * 1000, 150, 150)

    def test_zerteile_terminiert_und_ueberlappt(self):
        self.assertEqual(rag.zerteile("abcdefgh", 4, 2), ["abcd", "cdef", "efgh", "gh"])
        self.assertEqual(rag.zerteile("", 4, 2), [])
        self.assertEqual(rag.zerteile("abc", 800, 150), ["abc"])

    def test_load_chunks_prueft_die_konfiguration_zuerst(self):
        """Der Guard muss am Start greifen, nicht erst am ersten langen Text."""
        with mock.patch.object(rag, "CHUNK_SIZE", 150), \
             mock.patch.object(rag, "CHUNK_OVERLAP", 150):
            with self.assertRaises(rag.KonfigFehler):
                rag.load_chunks()

    def test_knowledge_chunks_haengen_nicht(self):
        with mock.patch.object(rag, "CHUNK_SIZE", 150), \
             mock.patch.object(rag, "CHUNK_OVERLAP", 150):
            with self.assertRaises(rag.KonfigFehler):
                rag.baue_knowledge_chunks([{"id": "k1", "seite": 2, "text": "A" * 400}], set())


# ---------------------------------------------------------------------------
# Der echte Durchlauf: cmd_eval verdrahtet die Wertung -- mit gestopften
# Modellaufrufen. Ohne diesen Test bliebe eine Mutation gruen, die cmd_eval an
# der reparierten Logik vorbei wieder auf die alte Formel legt.
# ---------------------------------------------------------------------------
class TestCmdEval(unittest.TestCase):
    def lauf(self, drop_types="", quelle="knowledge"):
        chunks = [{"doc": "knowledge", "seite": 1, "text": "x"}]
        nach_id = {f["frage"]: f["id"] for f in FRAGEN}

        def fake_retrieve(frage, _chunks, _embs, k=None):
            return [({"doc": "knowledge", "seite": s, "text": "x"}, 0.5)
                    for s in ABGERUFEN[nach_id[frage]]]

        puffer = io.StringIO()
        with mock.patch.object(rag, "build_index", lambda: (chunks, None)), \
             mock.patch.object(rag, "retrieve", fake_retrieve), \
             mock.patch.object(rag, "answer", lambda frage, hits: FAKTENFREI), \
             mock.patch.dict(os.environ, {"DROP_TYPES": drop_types, "SOURCE": quelle}), \
             contextlib.redirect_stdout(puffer):
            rag.cmd_eval(os.path.join(BASE, "golden_set.example.json"))
        return puffer.getvalue()

    def test_durchlauf_meldet_die_dreiteilung(self):
        text = self.lauf()
        self.assertIn("getroffen : 5/8", text)
        self.assertIn("verfehlt  : 3/8", text)
        self.assertIn("Verweigerungsfragen: 2/10", text)
        self.assertIn("Keyword-Regressionswarner: 1/10", text)
        self.assertIn("Verweigerungsfrage -- Retrieval-Quote nicht anwendbar", text)

    def test_durchlauf_ist_unabhaengig_von_drop_types(self):
        referenz = None
        for variante in DROP_VARIANTEN:
            for quelle in ("knowledge", "pdf"):
                with self.subTest(drop_types=variante, source=quelle):
                    zeilen = [z for z in self.lauf(variante, quelle).splitlines()
                              if "drop_types=" not in z and "source=" not in z]
                    if referenz is None:
                        referenz = zeilen
                    self.assertEqual(zeilen, referenz)


if __name__ == "__main__":
    unittest.main(verbosity=2)
