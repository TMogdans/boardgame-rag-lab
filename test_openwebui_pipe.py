#!/usr/bin/env python3
"""
Tests der Open-WebUI-Pipe -- laeuft OHNE Ollama und ohne Open WebUI.

    python test_openwebui_pipe.py     # oder: python -m unittest test_openwebui_pipe -v

Braucht pydantic und httpx (beide stecken im Open-WebUI-Image). Das Netz wird
ersetzt: Embeddings kommen aus einer festen Tabelle (requests.post gepatcht), das
LLM aus einem httpx.MockTransport, der mitschreibt, was Ollama bekommen haette.

Kern ist test_pipe_ruft_dieselbe_retrieval_wie_die_cli: die Pipe muss mit
denselben Stellschrauben dieselben Chunks in denselben Prompt legen wie rag.py.
"""
import asyncio, json, os, shutil, sys, tempfile, threading, time, unittest
from unittest import mock

import httpx

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
    {"id": 2, "seite": 11, "typ": "regel", "text": "Geld verdient man in Phase 5. Geld Geld. Danach folgt die Aufraeumphase mit allen Resten."},
    {"id": 3, "seite": 6, "typ": "regel", "text": "Die Kette wird am Anfang gebaut."},
    {"id": 4, "seite": 14, "typ": "regel", "text": "Geld aus der Bank, Kette zahlt."},
    {"id": 5, "seite": 9, "typ": "meta", "text": "Inhaltsverzeichnis"},
]


def sammle(agen):
    async def lauf():
        return [s async for s in agen]
    return asyncio.run(lauf())


def ollama_ndjson(*teile):
    return "".join(json.dumps({"message": {"content": t}, "done": False}) + "\n" for t in teile) + \
        json.dumps({"done": True}) + "\n"


class PipeTestBasis(unittest.TestCase):
    """Echte Pipe, echtes _stream -- nur das Netz ist ersetzt."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wissen = os.path.join(self.tmp.name, "knowledge.jsonl")
        self.schreibe_wissen(WISSEN)
        self.post = mock.patch("requests.post", side_effect=fake_post)
        self.post_mock = self.post.start()
        self.pipe = op.Pipe()
        self.pipe.valves = op.Pipe.Valves(RAG_DIR=BASE, KNOWLEDGE_PATH=self.wissen, OLLAMA_URL="http://stub",
                                          TOP_K=2, DROP_TYPES="flavor", CHUNK_SIZE=400, CHUNK_OVERLAP=150)
        # LLM: httpx.MockTransport statt Ollama
        self.llm_bekam = []
        self.llm_antwort = lambda req: httpx.Response(200, text=ollama_ndjson("Ant", "wort"))

        def handler(req):
            self.llm_bekam.append({"url": str(req.url), **json.loads(req.content)})
            return self.llm_antwort(req)
        echter_client = httpx.AsyncClient
        self.client = mock.patch.object(op.httpx, "AsyncClient",
                                        lambda **kw: echter_client(transport=httpx.MockTransport(handler), **kw))
        self.client.start()

    def tearDown(self):
        self.client.stop()
        self.post.stop()
        self.tmp.cleanup()

    def schreibe_wissen(self, eintraege):
        with open(self.wissen, "w") as f:
            for c in eintraege:
                f.write(json.dumps(c) + "\n")

    def frage(self, text, verlauf=(), task=None):
        body = {"messages": list(verlauf) + [{"role": "user", "content": text}]}
        return "".join(sammle(self.pipe.pipe(body, __task__=task)))

    def embed_aufrufe(self):
        return [c for c in self.post_mock.call_args_list if c.args[0].endswith("/api/embed")]

    def index_bauten(self):
        # Index-Bau = Embedding mit mehr als einem Text; eine Frage ist genau einer.
        return [c for c in self.embed_aufrufe() if len(c.kwargs["json"]["input"]) > 1]

    def nachrichten(self, i=0):
        return self.llm_bekam[i]["messages"]


class TestAequivalenzZurCli(PipeTestBasis):
    def test_pipe_ruft_dieselbe_retrieval_wie_die_cli(self):
        # Kleine Chunks, damit das Chunking wirklich etwas zerlegt und falsche
        # CHUNK_SIZE/OVERLAP-Uebernahme auffaellt.
        self.pipe.valves.CHUNK_SIZE, self.pipe.valves.CHUNK_OVERLAP = 30, 10
        frage = "Wie verdiene ich Geld?"
        text = self.frage(frage)

        # Referenz: CLI-Weg (rag.build_index) mit denselben Stellschrauben
        with mock.patch.dict(os.environ, {"SOURCE": "knowledge", "DROP_TYPES": "flavor"}), \
             mock.patch.multiple(rag, OLLAMA="http://stub", CHUNK_SIZE=30, CHUNK_OVERLAP=10), \
             mock.patch("os.path.dirname", return_value=self.tmp.name):
            chunks, embs = rag.build_index()
        self.assertGreater(len(chunks), len(WISSEN))  # es wurde tatsaechlich zerteilt
        hits = rag.retrieve(frage, chunks, embs, k=2)

        self.assertEqual(self.nachrichten(), rag.baue_nachrichten(frage, hits))
        erwartet = ", ".join(f"S. {h['seite']} ({s:.3f})" for h, s in hits)
        self.assertTrue(text.startswith("Antwort"), text)
        self.assertTrue(text.endswith(f"*Abgerufen: {erwartet}*"), text)

    def test_chunking_folgt_valves_nicht_der_container_umgebung(self):
        # Open WebUI benutzt CHUNK_SIZE/CHUNK_OVERLAP selbst -- das darf nicht durchschlagen.
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": "1000", "CHUNK_OVERLAP": "100", "RERANK": "1"}):
            self.pipe.valves.CHUNK_SIZE, self.pipe.valves.CHUNK_OVERLAP = 30, 10
            text = self.frage("Geld")
        # RERANK=1 in der Umgebung darf den Reranker (torch fehlt) nicht einschalten
        self.assertNotIn("Fehler", text)
        self.assertIn("*Abgerufen:", text)
        n_index = len(self.index_bauten()[0].kwargs["json"]["input"])
        erwartet = sum(len(rag.zerteile(c["text"], 30, 10)) for c in WISSEN if c["typ"] != "flavor")
        self.assertEqual(n_index, erwartet)


class TestValvesUndCache(PipeTestBasis):
    def test_embedding_nutzt_valves(self):
        self.pipe.valves.OLLAMA_URL, self.pipe.valves.EMBED_MODEL = "http://anderswo:1", "mein-embed"
        self.frage("Geld")
        for c in self.embed_aufrufe():
            self.assertEqual(c.args[0], "http://anderswo:1/api/embed")
            self.assertEqual(c.kwargs["json"]["model"], "mein-embed")

    def test_llm_nutzt_valves(self):
        self.pipe.valves.LLM_MODEL, self.pipe.valves.THINK = "mein-llm", True
        self.frage("Geld")
        b = self.llm_bekam[0]
        self.assertEqual(b["url"], "http://stub/api/chat")
        self.assertEqual((b["model"], b["think"], b["stream"]), ("mein-llm", True, True))

    def test_index_wird_einmal_gebaut(self):
        self.frage("Geld")
        self.frage("Kette")
        self.assertEqual(len(self.index_bauten()), 1)
        self.assertEqual(len(self.embed_aufrufe()), 3)  # 1x Index + 2x Frage

    def test_index_neu_bei_jeder_index_stellschraube(self):
        self.frage("Geld")
        for feld, wert in (("DROP_TYPES", "meta"), ("EMBED_MODEL", "x"), ("OLLAMA_URL", "http://y"),
                           ("CHUNK_OVERLAP", 20), ("CHUNK_SIZE", 50)):
            vorher = len(self.index_bauten())
            setattr(self.pipe.valves, feld, wert)
            self.frage("Geld")
            self.assertEqual(len(self.index_bauten()), vorher + 1, feld)

    def test_neubau_nutzt_die_neuen_werte(self):
        # Nicht nur zaehlen, dass neu gebaut wird -- sondern womit.
        self.frage("Geld")
        v = self.pipe.valves
        v.EMBED_MODEL, v.OLLAMA_URL, v.CHUNK_OVERLAP, v.CHUNK_SIZE = "neu-embed", "http://neu", 5, 20
        self.frage("Geld")
        bau = self.index_bauten()[-1]
        self.assertEqual(bau.args[0], "http://neu/api/embed")
        self.assertEqual(bau.kwargs["json"]["model"], "neu-embed")
        erwartet = sum(len(rag.zerteile(c["text"], 20, 5)) for c in WISSEN if c["typ"] != "flavor")
        self.assertEqual(len(bau.kwargs["json"]["input"]), erwartet)

    def test_parallele_anfragen_mit_wechselnden_valves_vermischen_keine_modelle(self):
        # Modell "m2" liefert Vektoren einer anderen Dimension. Wuerde eine Frage mit
        # dem Modell der Nachbaranfrage eingebettet, scheitert das Skalarprodukt.
        def post(url, json=None, timeout=None):
            if url.endswith("/api/embed"):
                extra = [1.0] if json["model"] == "m2" else []
                return FakeAntwort({"embeddings": [fake_vektor(t) + extra for t in json["input"]]})
            return fake_post(url, json=json, timeout=timeout)
        self.post_mock.side_effect = post
        self.frage("Geld")  # rag.py laden
        original = self.pipe._rag.retrieve

        def zoegernd(*a, **kw):
            time.sleep(0.005)  # Fenster, in dem eine Nachbaranfrage die Globalen umsetzen koennte
            return original(*a, **kw)
        self.pipe._rag.retrieve = zoegernd
        valves = [self.pipe.valves.model_copy(update={"EMBED_MODEL": m}) for m in ("m1", "m2")]
        fehler = []

        def lauf(v):
            for _ in range(10):
                try:
                    self.pipe._suche("Geld", v)
                except Exception as e:
                    fehler.append(repr(e))
        ts = [threading.Thread(target=lauf, args=(v,)) for v in valves]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(fehler, [])

    def test_kein_neubau_bei_antwort_stellschrauben(self):
        self.frage("Geld")
        self.pipe.valves.TOP_K, self.pipe.valves.LLM_MODEL, self.pipe.valves.THINK = 3, "z", True
        self.frage("Geld")
        self.assertEqual(len(self.index_bauten()), 1)

    def test_drop_types_wirkt(self):
        # Die Werbeseite 2 traegt "geld" auch -- sie darf trotzdem nie abgerufen werden.
        self.pipe.valves.TOP_K = 10
        text = self.frage("Geld")
        self.assertNotIn("S. 2 ", text)
        self.assertIn("S. 11 ", text)

    def test_drop_types_leer_nimmt_flavor_auf(self):
        self.pipe.valves.TOP_K, self.pipe.valves.DROP_TYPES = 10, ""
        self.assertIn("S. 2 ", self.frage("Geld"))

    def test_neue_wissensbasis_baut_index_neu(self):
        self.frage("Geld")
        self.schreibe_wissen(WISSEN + [{"id": 6, "seite": 15, "typ": "regel", "text": "Kette Kette Kette"}])
        self.assertIn("S. 15 ", self.frage("Kette"))

    def test_neuer_inhalt_mit_alter_mtime_baut_index_neu(self):
        self.frage("Geld")
        alt = os.stat(self.wissen).st_mtime_ns
        self.schreibe_wissen(WISSEN + [{"id": 6, "seite": 42, "typ": "regel", "text": "Kette Kette Kette"}])
        os.utime(self.wissen, ns=(alt, alt))  # wie cp -p / Restore
        self.assertIn("S. 42 ", self.frage("Kette"))

    def test_geaenderte_rag_py_wird_neu_geladen(self):
        rag_dir = os.path.join(self.tmp.name, "code")
        os.mkdir(rag_dir)
        shutil.copy(os.path.join(BASE, "rag.py"), rag_dir)
        self.pipe.valves.RAG_DIR = rag_dir
        self.frage("Geld")
        pfad = os.path.join(rag_dir, "rag.py")
        with open(pfad) as f:
            quelle = f.read()
        with open(pfad, "w") as f:
            f.write(quelle.replace("Du bist ein Regel-Assistent", "Du bist ein NEUER Regel-Assistent"))
        self.frage("Geld")
        self.assertTrue(self.nachrichten(1)[0]["content"].startswith("Du bist ein NEUER Regel-Assistent"))
        self.assertEqual(len(self.index_bauten()), 2)


class TestVerlaufUndTask(PipeTestBasis):
    def test_verlauf_geht_mit_quellen_nur_an_letzter_frage(self):
        verlauf = [{"role": "user", "content": "Wie verdiene ich Geld?"},
                   {"role": "assistant", "content": "In Phase 5."}]
        # Open WebUI schickt Nachrichten mit Anhang als Liste von Teilen
        roh = [{"role": "user", "content": [{"type": "text", "text": "Wie verdiene ich Geld?"}]}, verlauf[1]]
        self.frage("Und die Kette?", roh)
        n = self.nachrichten()
        self.assertEqual(n[0]["content"], rag.SYSTEM_PROMPT)
        self.assertEqual(n[1:3], verlauf)
        self.assertTrue(n[3]["content"].startswith("Quellen:"))
        self.assertTrue(n[3]["content"].endswith("Frage: Und die Kette?"))
        self.assertEqual(len(n), 4)

    def test_fusszeile_geht_nicht_in_den_verlauf(self):
        erste = self.frage("Wie verdiene ich Geld?")
        self.assertIn("*Abgerufen:", erste)
        verlauf = [{"role": "user", "content": "Wie verdiene ich Geld?"}, {"role": "assistant", "content": erste}]
        self.frage("Und die Kette?", verlauf)
        self.assertEqual(self.nachrichten(1)[2], {"role": "assistant", "content": "Antwort"})

    def test_task_ohne_retrieval_rollen_und_listen_erhalten(self):
        msgs = [{"role": "system", "content": "Du erzeugst Titel."},
                {"role": "user", "content": [{"type": "text", "text": "Erzeuge"}, {"type": "text", "text": "Titel"}]}]
        text = "".join(sammle(self.pipe.pipe({"messages": msgs}, __task__="title_generation")))
        self.assertEqual(text, "Antwort")
        self.assertEqual(self.embed_aufrufe(), [])
        self.assertEqual(self.nachrichten(), [{"role": "system", "content": "Du erzeugst Titel."},
                                              {"role": "user", "content": "Erzeuge Titel"}])

    def test_jede_task_art_ohne_retrieval(self):
        for task in ("title_generation", "tags_generation", "follow_up_generation"):
            self.frage("Erzeuge etwas", task=task)
        self.assertEqual(self.embed_aufrufe(), [])

    def test_abbruch_mitten_im_stream_ist_sauber(self):
        # Browser zu: Open WebUI schliesst den Generator. GeneratorExit darf nicht
        # als "Fehler" weiterverarbeitet werden (sonst RuntimeError beim Schliessen).
        async def lauf():
            agen = self.pipe.pipe({"messages": [{"role": "user", "content": "Geld"}]})
            erstes = await agen.__anext__()
            await agen.aclose()
            return erstes
        self.assertEqual(asyncio.run(lauf()), "Ant")

    def test_fehlerzeile_und_fusszeile_ohne_leerzeilen_nicht_im_verlauf(self):
        verlauf = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "---\n*Abgerufen: S. 11 (0.500)*"},
                   {"role": "user", "content": "b"}, {"role": "assistant", "content": "X\n\n**Fehler in der RAG-Pipe:** KonfigFehler: y"}]
        self.frage("c", verlauf)
        n = self.nachrichten()
        self.assertEqual([m["content"] for m in n[1:5]], ["a", "", "b", "X"])

    def test_leere_frage(self):
        self.assertEqual("".join(sammle(self.pipe.pipe({"messages": []}))), "Keine Frage gefunden.")


class TestFehlerSichtbar(PipeTestBasis):
    def test_konfigfehler_erscheint_im_chat(self):
        self.pipe.valves.DROP_TYPES = "fakt"
        text = self.frage("Geld")  # darf nicht werfen
        self.assertIn("Fehler in der RAG-Pipe", text)
        self.assertIn("KonfigFehler", text)
        self.assertIn("unbekannte Werte: fakt", text)

    def test_leerer_index_klar_gemeldet(self):
        self.schreibe_wissen([c for c in WISSEN if c["typ"] == "flavor"])
        self.assertIn("Index leer", self.frage("Geld"))

    def test_ollama_http_fehler_nennt_grund(self):
        self.llm_antwort = lambda req: httpx.Response(404, json={"error": "model 'qwen3:14b' not found"})
        text = self.frage("Geld")
        self.assertIn("404", text)
        self.assertIn("model 'qwen3:14b' not found", text)

    def test_ollama_fehler_mitten_im_stream(self):
        self.llm_antwort = lambda req: httpx.Response(
            200, text=json.dumps({"message": {"content": "Teil 1 "}}) + "\n" + json.dumps({"error": "runner terminated"}) + "\n")
        text = self.frage("Geld")
        self.assertTrue(text.startswith("Teil 1 "))
        self.assertIn("runner terminated", text)
        self.assertNotIn("*Abgerufen:", text)


class TestEventLoop(PipeTestBasis):
    def test_langsames_embedding_blockiert_den_event_loop_nicht(self):
        def langsam(url, json=None, timeout=None):
            time.sleep(0.3)
            return fake_post(url, json=json, timeout=timeout)
        self.post_mock.side_effect = langsam

        async def lauf():
            luecken, stop = [], asyncio.Event()

            async def herzschlag():
                t = time.monotonic()
                while not stop.is_set():
                    await asyncio.sleep(0.01)
                    jetzt = time.monotonic()
                    luecken.append(jetzt - t)
                    t = jetzt
            hb = asyncio.create_task(herzschlag())
            body = {"messages": [{"role": "user", "content": "Geld"}]}
            await asyncio.gather(*[self._sammle(body) for _ in range(2)])
            stop.set()
            await hb
            return max(luecken)
        # Blockierend waeren es >= 0,3 s (ein Embedding-Aufruf); frei bleibt es im Bereich von 10 ms.
        self.assertLess(asyncio.run(lauf()), 0.15)

    async def _sammle(self, body):
        return [s async for s in self.pipe.pipe(body)]


class TestRegelfrageFeld(PipeTestBasis):
    """Das optionale Feld "regelfrage" fuer Home Assistant."""

    def frage_rf(self, text, **rf):
        body = {"messages": [{"role": "user", "content": text}], "regelfrage": rf}
        return "".join(sammle(self.pipe.pipe(body)))

    def test_bekanntes_spiel_antwortet_wie_ohne_feld(self):
        mit = self.frage_rf("Geld", spiel="Food Chain Magnate")
        ohne = self.frage("Geld")
        self.assertEqual(mit, ohne)
        self.assertEqual(self.nachrichten(0), self.nachrichten(1))

    def test_alias_und_hoerfehler_treffen(self):
        for spiel in ("FCM", "food chain", "Food Chain Magnet"):
            self.assertIn("*Abgerufen:", self.frage_rf("Geld", spiel=spiel), spiel)

    def test_unbekanntes_spiel_ohne_suche(self):
        text = self.frage_rf("Geld", spiel="Terraforming Mars")
        self.assertIn("Zu „Terraforming Mars“ habe ich kein Regelheft", text)
        self.assertIn("Verfuegbar: Food Chain Magnate", text)
        self.assertEqual(self.embed_aufrufe(), [])
        self.assertEqual(self.llm_bekam, [])

    def test_aehnliches_spiel_bekommt_vorschlag(self):
        text = self.frage_rf("Geld", spiel="Fudschein Magnat")
        self.assertIn("Meintest du Food Chain Magnate?", text)
        self.assertEqual(self.embed_aufrufe(), [])

    def test_sprache_ohne_fusszeile_gleicher_prompt(self):
        text = self.frage_rf("Geld", spiel="FCM", sprache=True)
        self.assertEqual(text, "Antwort")
        self.frage("Geld")
        self.assertEqual(self.nachrichten(0), self.nachrichten(1))  # Prompt identisch zur Chat-Variante

    def test_spiel_aus_valves(self):
        self.pipe.valves.SPIEL, self.pipe.valves.SPIEL_ALIASE = "Brass: Birmingham", "Brass"
        self.assertIn("*Abgerufen:", self.frage_rf("Geld", spiel="brass birmingham"))
        self.assertIn("kein Regelheft", self.frage_rf("Geld", spiel="Food Chain Magnate"))

    def test_kaputtes_feld_wird_ignoriert(self):
        body = {"messages": [{"role": "user", "content": "Geld"}], "regelfrage": "Food Chain"}
        self.assertIn("*Abgerufen:", "".join(sammle(self.pipe.pipe(body))))


class TestSpielzuordnung(unittest.TestCase):
    K = op.katalog_aus("Food Chain Magnate", "Food Chain, FCM")

    def test_treffer(self):
        for q in ("Food Chain Magnate", "food-chain magnate", "FoodChain", "FCM", "Food Chain Magnet"):
            self.assertEqual(op.ordne_spiel(q, self.K), ("treffer", "Food Chain Magnate"), q)

    def test_normalisierung_traegt_bei_kurzen_namen(self):
        # Bei kurzen Namen rettet die Unschaerfe nichts: "f.c.m." gegen "fcm" liegt
        # ohne Normalisierung bei 0,67 -- erst das Entfernen der Satzzeichen trifft.
        self.assertEqual(op.ordne_spiel("F.C.M.", self.K), ("treffer", "Food Chain Magnate"))
        self.assertEqual(op.ordne_spiel("F C M", self.K), ("treffer", "Food Chain Magnate"))

    def test_kein_treffer(self):
        for q in ("Terraforming Mars", "Brass Birmingham", "", None, "Food"):
            self.assertEqual(op.ordne_spiel(q, self.K)[0], "unbekannt", q)

    def test_vorschlag_nur_bei_aehnlichkeit(self):
        self.assertEqual(op.ordne_spiel("Fudschein Magnat", self.K), ("unbekannt", ["Food Chain Magnate"]))
        self.assertEqual(op.ordne_spiel("Terraforming Mars", self.K), ("unbekannt", []))


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

    def test_zerlege_verlauf_nimmt_letzte_nutzernachricht_ohne_system(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}, {"role": "user", "content": " c "}]
        frage, verlauf = op.zerlege_verlauf(msgs)
        self.assertEqual(frage, "c")
        self.assertEqual(verlauf, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])

    def test_defaults_sind_die_gemessene_konfiguration(self):
        # README, Tabelle "Gemessen mit dem reparierten Stand": 400, top_k=4, flavor,meta
        v = op.Pipe.Valves()
        self.assertEqual((v.CHUNK_SIZE, v.CHUNK_OVERLAP, v.TOP_K, v.DROP_TYPES, v.EMBED_MODEL, v.LLM_MODEL, v.THINK),
                         (400, 150, 4, "flavor,meta", "bge-m3", "qwen3:14b", False))
        self.assertEqual((rag.CHUNK_OVERLAP, rag.EMBED_MODEL, rag.LLM_MODEL), (150, "bge-m3", "qwen3:14b"))

    def test_top_k_mindestens_eins(self):
        with self.assertRaises(Exception):
            op.Pipe.Valves(TOP_K=0)

    def test_ollama_zeile(self):
        self.assertEqual(op.ollama_zeile('{"message": {"content": "x"}}'), "x")
        self.assertEqual(op.ollama_zeile(""), "")
        with self.assertRaises(RuntimeError):
            op.ollama_zeile('{"error": "model not found"}')


if __name__ == "__main__":
    unittest.main()
