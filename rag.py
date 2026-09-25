#!/usr/bin/env python3
"""
Schlankes RAG-Labor fuer Brettspiel-Regelfragen.

Alle PDFs im Ordner pdfs/ werden automatisch indiziert -> neues Spiel = PDF reinlegen.
Die Stellschrauben (Chunking, Embedding-Modell, top_k, Reranker) stehen oben in der
Konfiguration und sind der eigentliche Gegenstand des Experiments. Alle per env setzbar.

Nutzung:
    python rag.py ask "Wie verdiene ich Geld?"   # eine Frage
    python rag.py eval                            # Golden Set durchlaufen
    RERANK=1 CHUNK_SIZE=400 python rag.py eval    # mit Reranker + kleineren Chunks

Grundsatz der Auswertung: Die Messlatte haengt an der Natur der Frage, nie an der
Konfiguration des Laufs. Keine Stellschraube (DROP_TYPES, SOURCE, CHUNK_SIZE) darf
einen Nenner verschieben -- sonst zieht dieselbe Einstellung, die das Retrieval
veraendert, auch den Beobachtungspunkt mit.
"""
import sys, json, glob, re, os
import numpy as np
import requests
# pypdf wird erst im PDF-Zweig von load_chunks importiert. Die Wissensbasis-Route
# (SOURCE=knowledge) und die komplette Wertungslogik brauchen es nicht -- ohne
# Top-Level-Import laesst sich die Auswertung ohne Ingestion-Abhaengigkeiten
# und ohne laufende Modelle testen (siehe test_wertung.py).

# ---------- Konfiguration: hier drehen wir fuer das Experiment (alles per env) ----------
OLLAMA        = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "bge-m3")   # deutsches/multilinguales Embedding
LLM_MODEL     = os.environ.get("LLM_MODEL", "qwen3:14b")  # Modell fix halten, Retrieval variieren
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 800))    # Zeichen pro Chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150)) # Zeichen Ueberlappung zwischen Chunks
TOP_K         = int(os.environ.get("TOP_K", 4))           # wie viele Chunks in den Kontext wandern
THINK         = os.environ.get("THINK", "0") == "1"       # Qwen3-Reasoning an/aus (langsamer, gruendlicher)
RERANK        = os.environ.get("RERANK", "0") == "1"      # zweite Stufe: Cross-Encoder-Reranking
RERANK_MODEL  = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CANDIDATES    = int(os.environ.get("CANDIDATES", 20))     # so viele grob abrufen, bevor der Reranker auf TOP_K eindampft
PDF_DIR       = os.path.join(os.path.dirname(__file__), "pdfs")

# Vokabular des Klassifikators (classify.py): nur diese drei Chunk-Typen werden
# ueberhaupt vergeben. DROP_TYPES darf nichts anderes nennen -- die Frage-Typen des
# Golden Sets (fakt/falle/flavor/leerstelle/tabelle) sind eine ANDERE Taxonomie,
# Schnittmenge ist allein "flavor". Genau diese Verwechslung hat einmal einen
# Nenner verschoben.
CHUNK_TYPEN = ("regel", "flavor", "meta")

SYSTEM_PROMPT = """Du bist ein Regel-Assistent, der Fragen ausschliesslich auf Basis der dir bereitgestellten Quellen (Regelwerke) beantwortet.

Regeln:
1. Antworte immer auf Deutsch.
2. Stuetze jede sachliche Aussage ausschliesslich auf die bereitgestellten Quellen. Verlasse dich fuer konkrete Werte (Zahlen, Betraege, Kartennamen, Dauer, Reichweiten) NIEMALS auf dein Gedaechtnis - uebernimm sie woertlich aus der Quelle.
3. Nenne zu jeder Aussage die Fundstelle (Seite). Pruefe VOR jeder Aussage, ob die zitierte Stelle sie wirklich stuetzt. Wenn nicht, triff die Aussage nicht.
4. Wenn mehrere aehnliche Elemente existieren (z.B. verschiedene Karten), vergewissere dich, dass Wert UND Name aus derselben Fundstelle stammen. Ordne eine Zahl nie der falschen Karte zu.
5. Interpretiere jede Frage im Kontext der bereitgestellten Dokumente (es geht um ein Brettspiel, nicht um die reale Welt).
6. Ist eine Frage nicht durch die Quellen gedeckt, sage ausdruecklich "Dazu enthaelt das Dokument keine Angaben." und rate NICHT.
7. Kein externes Wissen einbauen.
8. Lieber knapp und korrekt als ausfuehrlich und unsicher."""


class KonfigFehler(RuntimeError):
    """Die Lauf-Konfiguration ist in sich widerspruechlich.

    Lieber laut abbrechen als still eine Zahl produzieren, die etwas anderes
    misst als ihr Etikett behauptet.
    """


# ---------- Guards ----------
def pruefe_chunk_konfiguration(size=None, overlap=None):
    """CHUNK_SIZE muss echt groesser als CHUNK_OVERLAP sein.

    Sonst ist die Schrittweite <= 0 und die Chunk-Schleife kommt nie voran
    (gemessen bei CHUNK_SIZE=150 gegen den festen Overlap 150: start waechst
    nicht mehr, der Prozess haengt ohne Fehlermeldung).
    """
    size = CHUNK_SIZE if size is None else size
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    if size <= overlap:
        raise KonfigFehler(
            f"CHUNK_SIZE={size} muss groesser als CHUNK_OVERLAP={overlap} sein. "
            f"Schrittweite waere {size - overlap} (<= 0) -> Endlosschleife beim Chunken. "
            f"Also CHUNK_SIZE erhoehen oder CHUNK_OVERLAP senken."
        )
    return size, overlap


def zerteile(text, size=None, overlap=None):
    """Text in ueberlappende Stuecke schneiden. Prueft die Schrittweite an der Schleife selbst."""
    size, overlap = pruefe_chunk_konfiguration(size, overlap)
    schritt = size - overlap
    return [text[start:start + size] for start in range(0, len(text), schritt)]


def chunk_seite(c):
    """Seitenzahl eines knowledge.jsonl-Eintrags -- ohne stillen Fallback.

    Frueher stand hier c.get("seite", c["id"]): fehlte das Feld, wanderte die
    Chunk-ID in das Feld "seite" und der Systemprompt druckte sie als
    "Fundstelle (Seite)" aus. Gemessen auf einer ingest.py-Basis mit 79
    Eintraegen: Seitenangaben bis 79 bei einem 15-seitigen Heft.
    """
    if "seite" not in c:
        raise KonfigFehler(
            f"knowledge.jsonl-Eintrag {c.get('id', '?')!r} hat kein Feld 'seite'. "
            "Ohne Seite gibt es keine zitierfaehige Fundstelle -- frueher rutschte "
            "hier die Chunk-ID ins Feld 'seite' und wurde als Seitenzahl ausgegeben. "
            "Die Datei muss von einem Skript geschrieben sein, das 'seite' mitschreibt: "
            "auto_ingest.py tut das in allen Zweigen; ingest.py und vision_ingest.py "
            "muessen es ebenfalls tun. Kein Fallback, weil eine erfundene Seitenzahl "
            "schlimmer ist als ein Abbruch."
        )
    return c["seite"]


def lies_drop_types(env=None):
    """DROP_TYPES lesen und validieren -- oder hart abbrechen.

    DROP_TYPES ist ein INDEX-Filter (welche Chunks gar nicht erst eingebettet
    werden). Auf die Wertung wirkt es bewusst nirgends mehr. Damit es nicht
    still ins Leere greift, wird jede Konstellation abgelehnt, in der es gesetzt
    ist, aber nicht wirken kann.
    """
    env = os.environ if env is None else env
    drop = [x.strip() for x in env.get("DROP_TYPES", "").split(",") if x.strip()]
    if not drop:
        return set()
    unbekannt = [x for x in drop if x not in CHUNK_TYPEN]
    if unbekannt:
        raise KonfigFehler(
            f"DROP_TYPES nennt unbekannte Werte: {', '.join(unbekannt)}. "
            f"classify.py vergibt ausschliesslich {', '.join(CHUNK_TYPEN)}. "
            "Die Frage-Typen des Golden Sets (fakt, falle, leerstelle, tabelle) sind "
            "eine andere Taxonomie und hier nicht zulaessig."
        )
    quelle = env.get("SOURCE")
    if quelle != "knowledge":
        raise KonfigFehler(
            f"DROP_TYPES={','.join(drop)} gesetzt, aber SOURCE={quelle!r}. "
            "Der Typ-Filter greift nur auf knowledge.jsonl (SOURCE=knowledge); "
            "auf dem pypdf-Weg gibt es keine Typen und der Filter wuerde nichts tun. "
            "Also SOURCE=knowledge setzen oder DROP_TYPES weglassen."
        )
    return set(drop)


def pruefe_typ_feld(rohchunks, drop):
    """DROP_TYPES ohne vorherigen classify.py-Lauf ist ein Irrtum, kein No-op."""
    if drop and not any("typ" in c for c in rohchunks):
        raise KonfigFehler(
            f"DROP_TYPES={','.join(sorted(drop))} gesetzt, aber kein Eintrag in "
            "knowledge.jsonl hat ein Feld 'typ' -- classify.py zuerst laufen lassen. "
            "Ohne Typen filtert der Filter nichts und die Zahl waere eine andere, "
            "als das Etikett behauptet."
        )


# ---------- PDF -> Chunks (seitenbewusst, damit Zitate eine Seite haben) ----------
def baue_knowledge_chunks(rohchunks, drop):
    pruefe_typ_feld(rohchunks, drop)
    chunks = []
    for c in rohchunks:
        if c.get("typ") in drop:   # z.B. DROP_TYPES="flavor,meta"
            continue
        seite = chunk_seite(c)
        for stueck in zerteile(c["text"]):
            chunks.append({"doc": "knowledge", "seite": seite, "text": stueck})
    return chunks


def load_chunks():
    pruefe_chunk_konfiguration()
    # DROP_TYPES unbedingt validieren, auch auf dem PDF-Weg: die Variable wurde
    # bisher immer gelesen, wirkte aber nur bei SOURCE=knowledge.
    drop = lies_drop_types()
    # Verbalisierte Wissensbasis (Docling -> Qwen) statt roher pypdf-Extraktion?
    if os.environ.get("SOURCE") == "knowledge":
        with open(os.path.join(os.path.dirname(__file__), "knowledge.jsonl")) as kf:
            rohchunks = [json.loads(line) for line in kf if line.strip()]
        return baue_knowledge_chunks(rohchunks, drop)
    from pypdf import PdfReader
    chunks = []
    for path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        doc = os.path.basename(path)
        for pageno, page in enumerate(PdfReader(path).pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                continue
            for stueck in zerteile(text):
                chunks.append({"doc": doc, "seite": pageno, "text": stueck})
    return chunks


# ---------- Embeddings via Ollama ----------
def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=np.float32)


def l2norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def build_index():
    chunks = load_chunks()
    embs = l2norm(embed([c["text"] for c in chunks]))
    return chunks, embs


# ---------- Reranker (lazy geladen) ----------
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, trust_remote_code=True)
    return _reranker


def retrieve(query, chunks, embs, k=TOP_K):
    q = l2norm(embed([query]))[0]
    sims = embs @ q
    if RERANK:
        # 1. Stufe: grob CANDIDATES per Embedding holen
        cand = [int(i) for i in np.argsort(-sims)[:CANDIDATES]]
        # 2. Stufe: Cross-Encoder bewertet jedes Frage-Chunk-Paar einzeln
        scores = get_reranker().predict([[query, chunks[i]["text"]] for i in cand])
        ranked = sorted(zip(cand, scores), key=lambda x: -x[1])[:k]
        return [(chunks[i], float(s)) for i, s in ranked]
    order = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in order]


# ---------- Antwort vom LLM ----------
def baue_nachrichten(query, hits):
    """Systemprompt + Quellen + Frage -- die eine Stelle, an der der Prompt entsteht.

    Die CLI (answer) und die Open-WebUI-Pipe (openwebui_pipe.py) bauen ihn beide
    hier, damit das Golden Set auch das misst, was im Chat ankommt.
    """
    kontext = "\n\n".join(f"[{h['doc']}, Seite {h['seite']}]\n{h['text']}" for h, _ in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Quellen:\n{kontext}\n\nFrage: {query}"},
    ]


def answer(query, hits):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM_MODEL,
        "messages": baue_nachrichten(query, hits),
        "think": THINK,
        "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------- Wertung ----------
KAT_GETROFFEN    = "getroffen"
KAT_VERFEHLT     = "verfehlt"
KAT_VERWEIGERUNG = "verweigerung"


def _keyword_muster(kw):
    r"""Regex fuer ein Keyword mit Wortgrenzen, soweit die Randzeichen Wortzeichen sind.

    Reines Substring-Matching hat gemessen Falsch-Positive erzeugt: "nicht" steckt
    in "nichts". \b laesst sich aber nicht blind anhaengen -- Keywords wie "50 %"
    oder "#12" beginnen bzw. enden mit Nicht-Wortzeichen, dort wuerde \b nie passen.
    Deshalb Lookarounds nur an den Seiten, an denen ein Wortzeichen steht.
    """
    links  = r"(?<!\w)" if kw[:1].isalnum() or kw[:1] == "_" else ""
    rechts = r"(?!\w)"  if kw[-1:].isalnum() or kw[-1:] == "_" else ""
    # Deutsche Flexion: bei MEHRWORTIGEN Keywords darf das letzte Wort eine Endung
    # tragen ("keine Angabe" trifft auch "keine Angaben"). Gemessen aufgefallen:
    # das Modell verweigert mit "keine Angaben", das Golden Set nennt "keine Angabe",
    # und die reine Wortgrenze erzeugte dadurch ein Falsch-NEGATIV.
    # Bei EINWORTIGEN Keywords bleibt es strikt -- sonst traefe "nicht" wieder
    # "nichts", genau das Falsch-Positiv, das die Wortgrenze beseitigen sollte.
    # Endet das Keyword auf einer Ziffer, bleibt es ebenfalls strikt: Zahlen
    # flektieren nicht, und "2 bis 5" darf nicht "2 bis 555" treffen.
    if rechts and len(kw.split()) > 1 and kw[-1:].isalpha():
        rechts = r"\w{0,3}(?!\w)"
    return links + re.escape(kw) + rechts


def keyword_treffer(keywords, antwort):
    """Welche erwarteten Stichwoerter stehen woertlich (auf Wortgrenze) in der Antwort?

    ACHTUNG, Grundsatz: Das ist ein REGRESSIONSWARNER, kein Korrektheitsmass.
    Der Warner sagt, ob ein erwartetes Stichwort vorkommt. Er kann nicht sagen,
    ob eine Antwort richtig ist -- und schon gar nicht, ob eine falsch ist. Eine
    faktenfreie Antwort kann Stichwoerter treffen, eine korrekt umformulierte
    Antwort kann sie verfehlen. Wer aus dieser Zahl "keine einzige falsch" liest,
    liest etwas, das dieses Skript nicht erhebt.
    """
    return [k for k in (keywords or []) if k and re.search(_keyword_muster(k), antwort, re.IGNORECASE)]


def bewerte_retrieval(frage, abgerufene_seiten):
    """Dreiteilung: getroffen / verfehlt / Verweigerungsfrage.

    Die Zuordnung haengt AUSSCHLIESSLICH an der Frage selbst -- nie an DROP_TYPES,
    SOURCE, CHUNK_SIZE oder irgendeiner anderen Stellschraube des Laufs. Vorher
    hing sie am Frage-Typ gegen DROP_TYPES; gemessen hat DROP_TYPES=falle damit
    die Quote von 5/9 auf 5/7 gehoben, ohne einen einzigen Chunk aus dem Index zu
    nehmen, und DROP_TYPES=fakt,falle,flavor,tabelle meldete 0/0 ohne Warnung.

    Verweigerungsfragen (erwartet_verweigerung: true im Golden Set) sind eine
    eigene Kategorie, kein stiller Abzug vom Nenner: bei ihnen ist Verweigern die
    richtige Antwort, eine Retrieval-Quote ist auf sie nicht anwendbar.
    """
    if frage.get("erwartet_verweigerung"):
        return KAT_VERWEIGERUNG
    erwartete = frage.get("seiten") or []
    if not erwartete:
        raise KonfigFehler(
            f"Frage {frage.get('id', '?')} hat weder 'seiten' noch "
            "'erwartet_verweigerung': true. Damit ist unklar, was sie messen soll, "
            "und sie wuerde still aus dem Nenner fallen. Golden Set ergaenzen."
        )
    return KAT_GETROFFEN if any(s in abgerufene_seiten for s in erwartete) else KAT_VERFEHLT


def bewerte_frage(frage, abgerufene_seiten, antwort):
    """Ein Wertungssatz pro Frage. Liest keine Umgebungsvariablen."""
    return {
        "id": frage.get("id"),
        "typ": frage.get("typ"),
        "kategorie": bewerte_retrieval(frage, abgerufene_seiten),
        "erwartete_seiten": list(frage.get("seiten") or []),
        "abgerufene_seiten": list(abgerufene_seiten),
        "keywords_getroffen": keyword_treffer(frage.get("keywords"), antwort),
    }


def fasse_zusammen(saetze):
    """Aggregat mit ausgeschriebenen Nennern -- keine Quote ohne Bezugsgroesse."""
    verweigerung = [s for s in saetze if s["kategorie"] == KAT_VERWEIGERUNG]
    getroffen    = [s for s in saetze if s["kategorie"] == KAT_GETROFFEN]
    verfehlt     = [s for s in saetze if s["kategorie"] == KAT_VERFEHLT]
    return {
        "fragen": len(saetze),
        "retrieval_nenner": len(getroffen) + len(verfehlt),
        "getroffen": len(getroffen),
        "verfehlt": len(verfehlt),
        "verweigerungsfragen": len(verweigerung),
        "verweigerung_signal": sum(1 for s in verweigerung if s["keywords_getroffen"]),
        "kw_nenner": len(saetze),
        "kw_treffer": sum(1 for s in saetze if s["keywords_getroffen"]),
    }


def formatiere_zusammenfassung(z):
    """Beide Metriken mit explizitem Nenner und ehrlichem Etikett."""
    return [
        "== Zusammenfassung ==",
        f"Fragen im Golden Set: {z['fragen']}",
        f"  davon in der Retrieval-Wertung (erwartete Fundstelle vorhanden): {z['retrieval_nenner']}",
        f"  davon Verweigerungsfragen (Verweigern IST die richtige Antwort):  {z['verweigerungsfragen']}",
        "",
        f"Retrieval, erwartete Seite unter top_k={TOP_K}:",
        f"  getroffen : {z['getroffen']}/{z['retrieval_nenner']}",
        f"  verfehlt  : {z['verfehlt']}/{z['retrieval_nenner']}",
        f"  Nenner {z['retrieval_nenner']} haengt allein am Golden Set und aendert sich mit keiner Stellschraube.",
        "",
        f"Verweigerungsfragen: {z['verweigerungsfragen']}/{z['fragen']} -- nicht Teil der Retrieval-Quote.",
        f"  mit Verweigerungs-Signalwort in der Antwort: {z['verweigerung_signal']}/{z['verweigerungsfragen']}",
        "  Das ist ein Indiz, kein Urteil: ob tatsaechlich korrekt verweigert wurde,",
        "  entscheidet nur der Mensch beim Lesen der Antworten.",
        "",
        f"Keyword-Regressionswarner: {z['kw_treffer']}/{z['kw_nenner']} Fragen mit mindestens einem Stichworttreffer.",
        "  KEIN Korrektheitsmass. Der Warner prueft nur, ob ein erwartetes Stichwort",
        "  woertlich vorkommt. Er erkennt keine falsche Antwort und belegt kein",
        "  'keine einzige falsch' -- diese Kategorie erhebt das Skript nicht.",
    ]


# ---------- Modi ----------
def cmd_ask(query):
    chunks, embs = build_index()
    hits = retrieve(query, chunks, embs)
    print(answer(query, hits))
    print("\nAbgerufen:", [(h["doc"], f"S.{h['seite']}", round(s, 3)) for h, s in hits])


def cmd_eval(golden_set_pfad=None):
    # Pfad als Parameter, damit der komplette Wertungsdurchlauf mit einem
    # Beispiel-Golden-Set testbar ist (siehe test_wertung.py, TestCmdEval).
    gs = json.load(open(golden_set_pfad
                        or os.path.join(os.path.dirname(__file__), "golden_set.json")))
    chunks, embs = build_index()
    print(f"Config: chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}  top_k={TOP_K}  rerank={RERANK}"
          f"{'(' + RERANK_MODEL + ', cand=' + str(CANDIDATES) + ')' if RERANK else ''}"
          f"  embed={EMBED_MODEL}  llm={LLM_MODEL}  think={THINK}"
          f"  source={os.environ.get('SOURCE', 'pdf')}  drop_types={os.environ.get('DROP_TYPES', '') or '-'}")
    print(f"Index: {len(chunks)} Chunks aus {len(set(c['doc'] for c in chunks))} PDF(s)\n")
    saetze = []
    for f in gs["fragen"]:
        hits = retrieve(f["frage"], chunks, embs)
        ans = answer(f["frage"], hits)
        satz = bewerte_frage(f, [h["seite"] for h, _ in hits], ans)
        saetze.append(satz)
        print(f"[{f['id']}] ({f['typ']}) {f['frage']}")
        print(f"    erwartet : {f['erwartet']}")
        if satz["kategorie"] == KAT_VERWEIGERUNG:
            print(f"    Wertung  : Verweigerungsfrage -- Retrieval-Quote nicht anwendbar "
                  f"(abgerufen={satz['abgerufene_seiten']})")
        else:
            print(f"    Wertung  : {satz['kategorie']} (erwartet={satz['erwartete_seiten']} "
                  f"abgerufen={satz['abgerufene_seiten']})")
        print(f"    Keywords : {satz['keywords_getroffen'] or 'KEINE getroffen'}   (Regressionswarner)")
        print(f"    Antwort  : {ans[:280]}")
        print()
    for zeile in formatiere_zusammenfassung(fasse_zusammen(saetze)):
        print(zeile)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    try:
        if mode == "ask" and len(sys.argv) > 2:
            cmd_ask(" ".join(sys.argv[2:]))
        else:
            cmd_eval()
    except KonfigFehler as e:
        print(f"ABBRUCH (Konfiguration): {e}", file=sys.stderr)
        sys.exit(2)
