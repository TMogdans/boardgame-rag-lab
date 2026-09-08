#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests fuer classify.auswerten() -- laeuft ohne Ollama und ohne Modell.

    python test_classify.py

Der Fall-Katalog FAELLE ist die Messgrundlage aus der Fix-Liste (A4):
17 realistische Modellantworten. Er reproduziert beide dort veroeffentlichten
Zahlen (8 von 17 falsch mit der alten Code-Reihenfolge, 1 von 17 mit der
umgedrehten Reihenfolge) -- test_reihenfolge_* pinnt das fest, damit niemand
den Katalog spaeter unbemerkt entschaerft.
"""
import unittest

import classify

# (Modellantwort, korrekte Kategorie)
FAELLE = [
    # --- die acht Faelle, die die alte Substring-Auswertung falsch loeste ---
    ("Das ist eine regel, kein flavor.", "regel"),
    ("Das ist kein flavor, sondern eine regel.", "regel"),
    ("Eindeutig eine Regel - kein Flavor-Text, keine Meta-Angabe.", "regel"),
    ("Es handelt sich um eine Spielregel und nicht um Flavor.", "regel"),
    ("Antwort: regel (der Abschnitt beschreibt eine Mechanik, nicht nur Flavor)", "regel"),
    ("Weder flavor noch meta: regel.", "regel"),
    ("regel, kein meta", "regel"),
    ("<think>Ist das flavor oder regel? ...</think>\nregel", "regel"),
    # --- einwortige und kurze Antworten (der vom Prompt verlangte Normalfall) ---
    ("regel", "regel"),
    ("flavor", "flavor"),
    ("meta", "meta"),
    ("Kategorie: flavor", "flavor"),
    ("Ich wuerde sagen: Spielregel.", "regel"),
    # --- der Fall, an dem die bloss umgedrehte Reihenfolge scheitert ---
    ("Kein Regelinhalt, sondern flavor.", "flavor"),
    # --- weitere Antworten, die schon vorher richtig liefen ---
    ("Inhaltsverzeichnis der Anleitung, also meta.", "meta"),
    ("Das ist reine Werbung: flavor.", "flavor"),
    ("meta (Impressum)", "meta"),
]


def alte_auswertung(antwort, reihenfolge):
    """Die Substring-Auswertung aus classify.py vor dem Fix (HEAD 703e782:31-35)."""
    a = antwort.strip().lower()
    for t in reihenfolge:
        if t in a:
            return t
    return "regel"


def falschzahl(funktion):
    return sum(1 for antwort, soll in FAELLE if funktion(antwort) != soll)


class TestKatalog(unittest.TestCase):
    def test_katalog_hat_17_faelle(self):
        self.assertEqual(len(FAELLE), 17)

    def test_reihenfolge_code_loest_8_von_17_falsch(self):
        # Referenzmessung, keine Anforderung an den neuen Code.
        self.assertEqual(
            falschzahl(lambda a: alte_auswertung(a, ("flavor", "meta", "regel"))), 8)

    def test_reihenfolge_umgedreht_loest_1_von_17_falsch(self):
        # Belegt: blosses Umdrehen der Reihenfolge reicht nicht.
        self.assertEqual(
            falschzahl(lambda a: alte_auswertung(a, ("regel", "meta", "flavor"))), 1)


class TestAuswerten(unittest.TestCase):
    def test_alle_17_faelle(self):
        falsch = [(a, classify.auswerten(a), soll)
                  for a, soll in FAELLE if classify.auswerten(a) != soll]
        self.assertEqual(falsch, [], f"{len(falsch)} von {len(FAELLE)} falsch: {falsch}")

    def test_einwortig_greift_zuerst(self):
        # Der Prompt verlangt EIN Wort; dieser Weg darf nicht an Heuristik haengen.
        for kat in ("regel", "flavor", "meta"):
            self.assertEqual(classify.auswerten(kat), kat)
            self.assertEqual(classify.auswerten(f"  {kat.upper()}\n"), kat)

    def test_think_block_wird_entfernt(self):
        self.assertEqual(
            classify.auswerten("<think>flavor? meta? nein.</think>\nregel"), "regel")
        self.assertEqual(
            classify.auswerten("<think>regel oder meta?</think>\nflavor"), "flavor")
        # unabgeschlossener think-Block (abgeschnittene Antwort)
        self.assertEqual(classify.auswerten("<think>vielleicht flavor"), "regel")

    def test_letzte_zeile_entscheidet(self):
        self.assertEqual(
            classify.auswerten("Ueberlegung: flavor waere denkbar.\nmeta"), "meta")

    def test_wortgrenzen_statt_substring(self):
        # "Spielregel"/"Regelinhalt" sind keine eigenstaendige Nennung von "regel",
        # "Flavor-Text" dagegen schon.
        self.assertEqual(classify.auswerten("Das ist reiner Flavor-Text."), "flavor")
        self.assertEqual(classify.auswerten("Enthaelt Regelinhalt."), "regel")

    def test_mehrere_kategorien_verwerfen_nicht(self):
        # Zweifelsfall: darf NIE flavor/meta liefern, sonst fliegt der Chunk
        # bei DROP_TYPES=flavor,meta aus dem Index.
        for antwort in ("flavor oder regel, schwer zu sagen",
                        "regel, flavor und meta kommen alle vor",
                        "Ist das flavor oder meta?"):
            self.assertEqual(classify.auswerten(antwort), "regel", antwort)

    def test_negierte_kategorie_zaehlt_nicht(self):
        self.assertEqual(classify.auswerten("kein flavor"), "regel")
        self.assertEqual(classify.auswerten("Das ist flavor, keine regel."), "flavor")

    def test_fallback_bei_unbrauchbarer_antwort(self):
        for antwort in ("", "   ", "Weiss ich nicht.", "Kategorie: unklar",
                        "Der Ausschnitt beschreibt den Spielaufbau."):
            self.assertEqual(classify.auswerten(antwort), "regel", repr(antwort))

    def test_nie_etwas_anderes_als_die_drei_kategorien(self):
        for antwort, _ in FAELLE:
            self.assertIn(classify.auswerten(antwort), ("regel", "flavor", "meta"))


class TestDocstringZusage(unittest.TestCase):
    """Die Zusage des Modul-Docstrings muss die Wirkung sein, nicht nur der Text."""

    def test_fallback_ist_regel(self):
        self.assertEqual(classify.FALLBACK, "regel")

    def test_keine_fehlklassifikation_in_richtung_verwerfen(self):
        # Die Richtung ist der eigentliche Schaden: eine Regel als flavor/meta
        # zu taggen loescht sie aus dem Index. Umgekehrt bleibt nur Rauschen drin.
        verworfen = [(a, classify.auswerten(a)) for a, soll in FAELLE
                     if soll == "regel" and classify.auswerten(a) != "regel"]
        self.assertEqual(verworfen, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
