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
"""
import sys, json, glob, re, os
import numpy as np
import requests
from pypdf import PdfReader

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


# ---------- PDF -> Chunks (seitenbewusst, damit Zitate eine Seite haben) ----------
def load_chunks():
    # Verbalisierte Wissensbasis (Docling -> Qwen) statt roher pypdf-Extraktion?
    if os.environ.get("SOURCE") == "knowledge":
        chunks = []
        with open(os.path.join(os.path.dirname(__file__), "knowledge.jsonl")) as kf:
            for line in kf:
                c = json.loads(line)
                t = c["text"]
                start = 0
                while start < len(t):
                    chunks.append({"doc": "knowledge", "seite": c.get("seite", c["id"]), "text": t[start:start + CHUNK_SIZE]})
                    start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
    chunks = []
    for path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        doc = os.path.basename(path)
        for pageno, page in enumerate(PdfReader(path).pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                chunks.append({"doc": doc, "seite": pageno, "text": text[start:start + CHUNK_SIZE]})
                start += CHUNK_SIZE - CHUNK_OVERLAP
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
def answer(query, hits):
    kontext = "\n\n".join(f"[{h['doc']}, Seite {h['seite']}]\n{h['text']}" for h, _ in hits)
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Quellen:\n{kontext}\n\nFrage: {query}"},
        ],
        "think": THINK,
        "stream": False,
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ---------- Modi ----------
def cmd_ask(query):
    chunks, embs = build_index()
    hits = retrieve(query, chunks, embs)
    print(answer(query, hits))
    print("\nAbgerufen:", [(h["doc"], f"S.{h['seite']}", round(s, 3)) for h, s in hits])


def cmd_eval():
    gs = json.load(open(os.path.join(os.path.dirname(__file__), "golden_set.json")))
    chunks, embs = build_index()
    print(f"Config: chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}  top_k={TOP_K}  rerank={RERANK}"
          f"{'(' + RERANK_MODEL + ', cand=' + str(CANDIDATES) + ')' if RERANK else ''}"
          f"  embed={EMBED_MODEL}  llm={LLM_MODEL}  think={THINK}")
    print(f"Index: {len(chunks)} Chunks aus {len(set(c['doc'] for c in chunks))} PDF(s)\n")
    seiten_treffer, kw_treffer, zaehlbar = 0, 0, 0
    gefilterte_typen = {x for x in os.environ.get("DROP_TYPES", "").split(",") if x}
    for f in gs["fragen"]:
        hits = retrieve(f["frage"], chunks, embs)
        ans = answer(f["frage"], hits)
        seiten = [h["seite"] for h, _ in hits]
        erwartete = f.get("seiten") or []
        # Fragt eine Frage nach genau dem Inhaltstyp, den DROP_TYPES aus dem Index
        # entfernt (z.B. typ="flavor" bei DROP_TYPES="flavor"), dann ist ihre erwartete
        # Fundstelle per Konfiguration nicht mehr auffindbar. Das ist weder Treffer noch
        # Fehlschlag des Retrievals -- sonst zaehlt der Filter als Retrieval-Fehler.
        # Die Seite allein genuegt als Pruefung nicht: auf der Deckblattseite stehen
        # neben dem Werbespruch weitere Chunks, die Seite bleibt also im Index.
        pruefbar = bool(erwartete) and f.get("typ") not in gefilterte_typen
        seite_ok = any(s in seiten for s in erwartete) if pruefbar else None
        kw = [k for k in f["keywords"] if k.lower() in ans.lower()]
        if pruefbar:
            zaehlbar += 1
            seiten_treffer += int(bool(seite_ok))
        kw_treffer += int(bool(kw))
        print(f"[{f['id']}] ({f['typ']}) {f['frage']}")
        print(f"    erwartet : {f['erwartet']}")
        grund = "" if pruefbar or not erwartete else "   (Typ per DROP_TYPES gefiltert -> nicht gewertet)"
        print(f"    Seite    : erwartet={erwartete} abgerufen={seiten} -> {seite_ok}{grund}")
        print(f"    Keywords : {kw if kw else 'KEINE getroffen'}")
        print(f"    Antwort  : {ans[:280]}")
        print()
    print(f"== Zusammenfassung ==")
    print(f"Retrieval (richtige Seite unter top_k): {seiten_treffer}/{zaehlbar}")
    print(f"Antwort-Keywords getroffen:            {kw_treffer}/{len(gs['fragen'])}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if mode == "ask" and len(sys.argv) > 2:
        cmd_ask(" ".join(sys.argv[2:]))
    else:
        cmd_eval()
