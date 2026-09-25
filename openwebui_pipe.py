"""
title: Brettspiel-Regeln (RAG)
description: Beantwortet Regelfragen aus knowledge.jsonl mit derselben Retrieval-Logik wie rag.py.
version: 0.2.0

Open-WebUI-Pipe: erscheint in der Modellauswahl als eigenes Modell. Die Logik
(Chunking, Embedding, Retrieval, Prompt) kommt aus rag.py -- diese Datei laedt
rag.py zur Laufzeit aus RAG_DIR und reicht nur durch. Mit denselben Werten fuer
CHUNK_SIZE, CHUNK_OVERLAP, DROP_TYPES und TOP_K misst `rag.py eval` also dieselbe
Pipeline, die im Chat antwortet.

Bewusst async: Open WebUI ruft eine synchrone Pipe direkt im Event-Loop auf,
eine blockierende Pipe wuerde die Oberflaeche fuer alle einfrieren, solange das
Modell rechnet. Die requests-basierten rag.py-Funktionen laufen deshalb per
asyncio.to_thread, die Antwort streamt ueber httpx.
"""
import asyncio
import importlib.util
import json
import os
import threading

import httpx
from pydantic import BaseModel, Field

FUSSZEILE_START = "\n\n---\n*Abgerufen: "


# ---------- reine Helfer (ohne Netz, testbar) ----------
def text_von(inhalt):
    """Nachrichteninhalt als Text -- Open WebUI schickt bei Anhaengen eine Liste von Teilen."""
    if isinstance(inhalt, list):
        return " ".join(t.get("text", "") for t in inhalt if isinstance(t, dict) and t.get("type") == "text")
    return inhalt or ""


def ohne_fusszeile(text):
    """Die eigene Fundstellen-Zeile aus einer frueheren Antwort entfernen.

    Open WebUI speichert sie als Teil der Assistant-Antwort. Ginge sie im Verlauf
    mit, saehe das Modell Seitenzahlen ohne deren Text -- der Systemprompt
    verlangt aber, nur bereitgestellte Quellen zu zitieren.
    """
    i = text.rfind(FUSSZEILE_START)
    return text[:i] if i >= 0 else text


def zerlege_verlauf(messages):
    """(frage, verlauf): letzte Nutzernachricht und die Wortwechsel davor.

    Retrieval laeuft nur auf der letzten Frage. Der Verlauf geht ohne seine alten
    Quellen mit, damit Rueckfragen ("und bei zwei Spielern?") verstanden werden.
    System-Nachrichten (Systemprompt aus Modell- oder Nutzereinstellungen) fallen
    weg: Der Regel-Systemprompt aus rag.py gilt allein.
    """
    letzte = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    if letzte is None:
        return "", []
    verlauf = []
    for m in messages[:letzte]:
        if m.get("role") == "user":
            verlauf.append({"role": "user", "content": text_von(m.get("content"))})
        elif m.get("role") == "assistant":
            verlauf.append({"role": "assistant", "content": ohne_fusszeile(text_von(m.get("content")))})
    return text_von(messages[letzte].get("content")).strip(), verlauf


def setze_verlauf_ein(nachrichten, verlauf):
    """Verlauf zwischen Systemprompt und die Quellen-Frage schieben."""
    return nachrichten[:1] + verlauf + nachrichten[1:]


def fundstellen(hits):
    """Fusszeile wie bei `rag.py ask`: welche Seiten wirklich im Kontext lagen."""
    teile = [f"S. {h['seite']} ({s:.3f})" for h, s in hits]
    return FUSSZEILE_START + ", ".join(teile) + "*"


def ollama_zeile(zeile):
    """Eine NDJSON-Zeile von /api/chat -> Textstueck ('' wenn keins)."""
    if not zeile.strip():
        return ""
    d = json.loads(zeile)
    if "error" in d:
        raise RuntimeError(f"Ollama: {d['error']}")
    return (d.get("message") or {}).get("content", "")


def datei_stand(pfad):
    """Aenderungsmerkmal einer Datei. mtime allein reicht nicht: `cp -p` oder ein
    Restore schreiben neuen Inhalt mit alter mtime."""
    st = os.stat(pfad)
    return (st.st_mtime_ns, st.st_size, st.st_ino)


# ---------- Pipe ----------
class Pipe:
    class Valves(BaseModel):
        RAG_DIR: str = Field("/rag/code", description="Ordner mit rag.py (Repo-Clone, read-only gemountet)")
        KNOWLEDGE_PATH: str = Field("/rag/data/knowledge.jsonl", description="Wissensbasis, gemountet")
        OLLAMA_URL: str = Field("http://ollama:11434", description="Ollama aus Sicht des Containers")
        EMBED_MODEL: str = "bge-m3"
        LLM_MODEL: str = "qwen3:14b"
        # Defaults = die in der README gemessene Konfiguration
        CHUNK_SIZE: int = Field(400, ge=1, description="Zeichen pro Chunk (wie CHUNK_SIZE bei rag.py)")
        CHUNK_OVERLAP: int = Field(150, ge=0, description="Ueberlappung (wie CHUNK_OVERLAP bei rag.py)")
        TOP_K: int = Field(4, ge=1)
        DROP_TYPES: str = Field("flavor,meta", description="Chunk-Typen, die nicht in den Index gehen (regel, flavor, meta)")
        THINK: bool = False

    def __init__(self):
        self.valves = self.Valves()
        self._lock = threading.Lock()
        self._rag = None
        self._rag_key = None
        self._index = None
        self._index_key = None

    # rag.py neu laden, wenn es sich geaendert hat (git pull im gemounteten Clone)
    def _lade_rag(self, v):
        pfad = os.path.join(v.RAG_DIR, "rag.py")
        key = (pfad, datei_stand(pfad))
        if key != self._rag_key:
            spec = importlib.util.spec_from_file_location("boardgame_rag", pfad)
            modul = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modul)
            self._rag, self._rag_key, self._index_key = modul, key, None
        # rag.py liest seine Konfiguration beim Import aus os.environ -- das ist hier
        # die Umgebung von Open WebUI, die CHUNK_SIZE/CHUNK_OVERLAP fuer ihre eigene
        # Dokumentsuche benutzt. Deshalb alles, was wirkt, explizit aus den Valves.
        rag = self._rag
        rag.OLLAMA = v.OLLAMA_URL
        rag.EMBED_MODEL = v.EMBED_MODEL
        rag.CHUNK_SIZE = v.CHUNK_SIZE
        rag.CHUNK_OVERLAP = v.CHUNK_OVERLAP
        rag.RERANK = False  # braucht torch, das im Open-WebUI-Image fehlt
        return rag

    def _lade_index(self, rag, v):
        key = (self._rag_key, v.KNOWLEDGE_PATH, datei_stand(v.KNOWLEDGE_PATH),
               v.DROP_TYPES, v.EMBED_MODEL, v.OLLAMA_URL, v.CHUNK_SIZE, v.CHUNK_OVERLAP)
        if key != self._index_key:
            # Gleiche Validierung wie die CLI mit SOURCE=knowledge
            drop = rag.lies_drop_types({"SOURCE": "knowledge", "DROP_TYPES": v.DROP_TYPES})
            with open(v.KNOWLEDGE_PATH) as f:
                roh = [json.loads(z) for z in f if z.strip()]
            chunks = rag.baue_knowledge_chunks(roh, drop)
            if not chunks:
                raise ValueError(f"Index leer: {v.KNOWLEDGE_PATH} enthaelt nach DROP_TYPES={v.DROP_TYPES!r} keinen Chunk.")
            embs = rag.l2norm(rag.embed([c["text"] for c in chunks]))
            self._index, self._index_key = (chunks, embs), key
        return self._index

    def _suche(self, frage):
        v = self.valves
        # Der Lock schuetzt nur Laden und Index-Bau; die Suche selbst (ein Embedding
        # der Frage) laeuft parallel.
        with self._lock:
            rag = self._lade_rag(v)
            chunks, embs = self._lade_index(rag, v)
        hits = rag.retrieve(frage, chunks, embs, k=v.TOP_K)
        return rag.baue_nachrichten(frage, hits), hits

    async def _stream(self, nachrichten):
        v = self.valves
        payload = {"model": v.LLM_MODEL, "messages": nachrichten, "think": v.THINK, "stream": True}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            async with client.stream("POST", f"{v.OLLAMA_URL}/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    try:
                        grund = json.loads(body).get("error", body)
                    except ValueError:
                        grund = body
                    raise RuntimeError(f"Ollama antwortet {r.status_code}: {grund}")
                async for zeile in r.aiter_lines():
                    stueck = ollama_zeile(zeile)
                    if stueck:
                        yield stueck

    async def pipe(self, body: dict, __task__=None):
        # Ein async-Generator wird von Open WebUI erst ausserhalb seines
        # Fehler-Handlers iteriert -- was hier entkommt, erreicht den Chat nicht.
        # Deshalb jeder Fehler als sichtbarer Text.
        try:
            async for stueck in self._antworte(body, __task__):
                yield stueck
        except Exception as e:
            yield f"\n\n**Fehler in der RAG-Pipe:** {type(e).__name__}: {e}"

    async def _antworte(self, body, task):
        messages = body.get("messages", [])
        if task:
            # Titel, Tags, Folgefragen: Open WebUI fragt dafuer das gewaehlte Modell.
            # Ohne Retrieval durchreichen -- sonst sucht die Pipe Regeln zu "Erzeuge einen Titel".
            async for stueck in self._stream([{"role": m["role"], "content": text_von(m.get("content"))}
                                              for m in messages]):
                yield stueck
            return
        frage, verlauf = zerlege_verlauf(messages)
        if not frage:
            yield "Keine Frage gefunden."
            return
        nachrichten, hits = await asyncio.to_thread(self._suche, frage)
        async for stueck in self._stream(setze_verlauf_ein(nachrichten, verlauf)):
            yield stueck
        yield fundstellen(hits)
