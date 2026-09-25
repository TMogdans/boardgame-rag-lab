#!/usr/bin/env python3
"""
Tests der Open-WebUI-Pipe -- laeuft OHNE Ollama und ohne Open WebUI.

    python test_openwebui_pipe.py     # oder: python -m unittest test_openwebui_pipe -v

Braucht pydantic und httpx (beide stecken im Open-WebUI-Image). Das Netz wird
ersetzt: Embeddings kommen aus einer festen Tabelle, das LLM aus einem Stub, der
mitschreibt, was es bekommen haette.

Kern ist test_pipe_ruft_dieselbe_retrieval_wie_die_cli: die Pipe muss dieselben
Chunks in denselben Prompt legen wie rag.py selbst.
"""
import asyncio, json, os, sys, tempfile, unittest
from unittest import mock

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import rag
import openwebui_pipe as op

# Feste "Embeddings": jede Achse ein Thema. Texte ohne Thema landen auf der Restachse.
ACHSEN = ("geld", "kette", "werbung", "rest")


def fake_vektor(text):
    t = text.lower()
    v = [float(t.count(a)) for a in ACHSEN[:-1]]
    return v + [0.1 if any(v) else 1.0]


class FakeAntwort:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def fake_post(url, json=None, timeout=None):
    if url.endswith("/api/embed"):
        return FakeAntwort({"embeddings": [fake_vektor(t) for t in json["input"]]})
    if url.endswith("/api/chat"):
        return FakeAntwort({"message": {"content": "CLI-Antwort"}})
    raise AssertionError(f"unerwartete URL {url}")


WISSEN = [
    {"id": 1, "seite": 2, "typ": "flavor", "text": "Werbung Werbung Geld! Das beste Spiel."},
    {"id": 2, "seite": 11, "typ": "regel", "text": "Geld verdient man in Phase 5. Geld Geld."},
    {"id": 3, "seite": 6, "typ": "regel", "text": "Die Kette wird am Anfang gebaut."},
    {"id": 4, "seite": 14, "typ": "regel", "text": "Geld aus der Bank, Kette zahlt."},
    {"id": 5, "seite": 9, "typ": "meta", "text": "Inhaltsverzeichnis"},
]


def sammle(agen):
    async def lauf():
        return [s async for s in agen]
    return asyncio.run(lauf())


class PipeTestBasis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wissen = os.path.join(self.tmp.name, "knowledge.jsonl")
        with open(self.wissen, "w") as f:
            for c in WISSEN:
                f.write(json.dumps(c) + "\n")
        self.post = mock.patch("requests.post", side_effect=fake_post)
        self.post_mock = self.post.start()
        self.pipe = op.Pipe()
        self.pipe.valves = op.Pipe.Valves(RAG_DIR=BASE, KNOWLEDGE_PATH=self.wissen,
                                          OLLAMA_URL="http://stub", TOP_K=2, DROP_TYPES="flavor")
        self.llm_bekam = []

        async def fake_stream(nachrichten):
            self.llm_bekam.append(nachrichten)
            for s in ("Ant", "wort"):
                yield s
        self.pipe._stream = fake_stream

    def tearDown(self):
        self.post.stop()
        self.tmp.cleanup()

    def frage(self, text, verlauf=()):
        body = {"messages": list(verlauf) + [{"role": "user", "content": text}]}
        return "".join(sammle(self.pipe.pipe(body)))

    def embed_aufrufe(self):
        return [c for c in self.post_mock.call_args_list if c.args[0].endswith("/api/embed")]


class TestPipe(PipeTestBasis):
    def test_pipe_ruft_dieselbe_retrieval_wie_die_cli(self):
        frage = "Wie verdiene ich Geld?"
        text = self.frage(frage)

        # Referenz: CLI-Weg mit derselben Konfiguration (SOURCE=knowledge, DROP_TYPES=flavor)
        with mock.patch.dict(os.environ, {"SOURCE": "knowledge", "DROP_TYPES": "flavor"}), \
             mock.patch.object(rag, "OLLAMA", "http://stub"), \
             mock.patch("os.path.dirname", return_value=self.tmp.name):
            chunks, embs = rag.build_index()
        hits = rag.retrieve(frage, chunks, embs, k=2)

        self.assertEqual(self.llm_bekam, [rag.baue_nachrichten(frage, hits)])
        erwartet = ", ".join(f"S. {h['seite']} ({s:.3f})" for h, s in hits)
        self.assertTrue(text.startswith("Antwort"))
        self.assertTrue(text.endswith(f"*Abgerufen: {erwartet}*"), text)

    def test_drop_types_wirkt(self):
        # Die Werbeseite 2 traegt "geld" auch -- sie darf trotzdem nie abgerufen werden.
        self.pipe.valves.TOP_K = 10
        text = self.frage("Geld")
        self.assertNotIn("S. 2 ", text)
        self.assertIn("S. 11 ", text)

    def test_drop_types_leer_nimmt_flavor_auf(self):
        # Gegenprobe zu test_drop_types_wirkt: ohne Filter taucht Seite 2 auf.
        self.pipe.valves.TOP_K = 10
        self.pipe.valves.DROP_TYPES = ""
        self.assertIn("S. 2 ", self.frage("Geld"))

    def test_unbekannter_drop_type_bricht_ab(self):
        self.pipe.valves.DROP_TYPES = "fakt"
        # Die Pipe laedt rag.py als eigenes Modul -> eigene KonfigFehler-Klasse.
        with self.assertRaisesRegex(Exception, "unbekannte Werte: fakt") as cm:
            self.frage("Geld")
        self.assertEqual(type(cm.exception).__name__, "KonfigFehler")

    def test_index_wird_einmal_gebaut(self):
        self.frage("Geld")
        self.frage("Kette")
        # 1x Index + 2x Frage
        self.assertEqual(len(self.embed_aufrufe()), 3)

    def test_neue_wissensbasis_baut_index_neu(self):
        self.frage("Geld")
        with open(self.wissen, "a") as f:
            f.write(json.dumps({"id": 6, "seite": 15, "typ": "regel", "text": "Kette Kette Kette"}) + "\n")
        os.utime(self.wissen, (1, 1))  # mtime sicher aendern
        self.assertIn("S. 15 ", self.frage("Kette"))

    def test_verlauf_geht_mit_quellen_nur_an_letzter_frage(self):
        verlauf = [{"role": "user", "content": "Wie verdiene ich Geld?"},
                   {"role": "assistant", "content": "In Phase 5."}]
        self.frage("Und die Kette?", verlauf)
        n = self.llm_bekam[0]
        self.assertEqual(n[0]["content"], rag.SYSTEM_PROMPT)
        self.assertEqual(n[1:3], verlauf)
        self.assertTrue(n[3]["content"].startswith("Quellen:"))
        self.assertTrue(n[3]["content"].endswith("Frage: Und die Kette?"))
        self.assertEqual(len(n), 4)

    def test_task_ohne_retrieval(self):
        body = {"messages": [{"role": "user", "content": "Erzeuge einen Titel"}]}
        text = "".join(sammle(self.pipe.pipe(body, __task__="title_generation")))
        self.assertEqual(text, "Antwort")
        self.assertEqual(self.embed_aufrufe(), [])
        self.assertEqual(self.llm_bekam, [[{"role": "user", "content": "Erzeuge einen Titel"}]])

    def test_leere_frage(self):
        self.assertEqual("".join(sammle(self.pipe.pipe({"messages": []}))), "Keine Frage gefunden.")


class TestHelfer(unittest.TestCase):
    def test_cli_answer_nutzt_baue_nachrichten(self):
        hits = [({"doc": "knowledge", "seite": 11, "text": "Geld in Phase 5"}, 0.9)]
        with mock.patch("requests.post", side_effect=fake_post) as p:
            rag.answer("Geld?", hits)
        self.assertEqual(p.call_args.kwargs["json"]["messages"], rag.baue_nachrichten("Geld?", hits))

    def test_text_von_liste(self):
        teile = [{"type": "text", "text": "Wie"}, {"type": "image_url", "image_url": {}},
                 {"type": "text", "text": "geht das?"}]
        self.assertEqual(op.text_von(teile), "Wie geht das?")

    def test_zerlege_verlauf_nimmt_letzte_nutzernachricht(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}, {"role": "user", "content": " c "}]
        frage, verlauf = op.zerlege_verlauf(msgs)
        self.assertEqual(frage, "c")
        self.assertEqual(verlauf, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])

    def test_ollama_zeile(self):
        self.assertEqual(op.ollama_zeile('{"message": {"content": "x"}}'), "x")
        self.assertEqual(op.ollama_zeile(""), "")
        with self.assertRaises(RuntimeError):
            op.ollama_zeile('{"error": "model not found"}')


if __name__ == "__main__":
    unittest.main()
