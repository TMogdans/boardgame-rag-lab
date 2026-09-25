"""
title: Brettspiel-Regeln (RAG)
description: Beantwortet Regelfragen aus knowledge.jsonl mit derselben Retrieval-Logik wie rag.py.
version: 0.1.0

Open-WebUI-Pipe: erscheint in der Modellauswahl als eigenes Modell. Die Logik
(Chunking, Embedding, Retrieval, Prompt) kommt aus rag.py -- diese Datei laedt
rag.py zur Laufzeit aus RAG_DIR und reicht nur durch. So misst das Golden Set
(rag.py eval) dieselbe Pipeline, die im Chat antwortet.

Einrichtung: README, Abschnitt "Open WebUI".

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


# ---------- reine Helfer (ohne Netz, testbar) ----------
def text_von(inhalt):
    """Nachrichteninhalt als Text -- Open WebUI schickt bei Anhaengen eine Liste von Teilen."""
    if isinstance(inhalt, list):
        return " ".join(t.get("text", "") for t in inhalt if isinstance(t, dict) and t.get("type") == "text")
    return inhalt or ""


def zerlege_verlauf(messages):
    """(frage, verlauf): letzte Nutzernachricht und die Wortwechsel davor.

    Retrieval laeuft nur auf der letzten Frage. Der Verlauf geht ohne seine alten
    Quellen mit, damit Rueckfragen ("und bei zwei Spielern?") verstanden werden.
    """
    letzte = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    if letzte is None:
        return "", []
    verlauf = [{"role": m["role"], "content": text_von(m.get("content"))}
               for m in messages[:letzte] if m.get("role") in ("user", "assistant")]
    return text_von(messages[letzte].get("content")).strip(), verlauf


def setze_verlauf_ein(nachrichten, verlauf):
    """Verlauf zwischen Systemprompt und die Quellen-Frage schieben."""
    return nachrichten[:1] + verlauf + nachrichten[1:]


def fundstellen(hits):
    """Fusszeile wie bei `rag.py ask`: welche Seiten wirklich im Kontext lagen."""
    teile = [f"S. {h['seite']} ({s:.3f})" for h, s in hits]
    return "\n\n---\n*Abgerufen: " + ", ".join(teile) + "*"


def ollama_zeile(zeile):
    """Eine NDJSON-Zeile von /api/chat -> Textstueck ('' wenn keins)."""
    if not zeile.strip():
        return ""
    d = json.loads(zeile)
    if "error" in d:
        raise RuntimeError(f"Ollama: {d['error']}")
    return (d.get("message") or {}).get("content", "")


# ---------- Pipe ----------
class Pipe:
    class Valves(BaseModel):
        RAG_DIR: str = Field("/rag/code", description="Ordner mit rag.py (Repo-Clone, read-only gemountet)")
        KNOWLEDGE_PATH: str = Field("/rag/data/knowledge.jsonl", description="Wissensbasis, gemountet")
        OLLAMA_URL: str = Field("http://ollama:11434", description="Ollama aus Sicht des Containers")
        EMBED_MODEL: str = "bge-m3"
        LLM_MODEL: str = "qwen3:14b"
        TOP_K: int = 4
        DROP_TYPES: str = Field("flavor", description="Chunk-Typen, die nicht in den Index gehen (regel, flavor, meta)")
        THINK: bool = False

    def __init__(self):
        self.valves = self.Valves()
        self._lock = threading.Lock()
        self._rag = None
        self._rag_key = None
        self._index = None
        self._index_key = None

    # rag.py neu laden, wenn es sich geaendert hat (git pull im gemounteten Clone)
    def _lade_rag(self):
        pfad = os.path.join(self.valves.RAG_DIR, "rag.py")
        key = (pfad, os.path.getmtime(pfad))
        if key != self._rag_key:
            spec = importlib.util.spec_from_file_location("boardgame_rag", pfad)
            modul = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modul)
            self._rag, self._rag_key, self._index_key = modul, key, None
        # rag.py liest diese Globals zur Aufrufzeit
        self._rag.OLLAMA = self.valves.OLLAMA_URL
        self._rag.EMBED_MODEL = self.valves.EMBED_MODEL
        return self._rag

    def _lade_index(self, rag):
        v = self.valves
        key = (self._rag_key, v.KNOWLEDGE_PATH, os.path.getmtime(v.KNOWLEDGE_PATH),
               v.DROP_TYPES, v.EMBED_MODEL, v.OLLAMA_URL)
        if key != self._index_key:
            # Gleiche Validierung wie die CLI mit SOURCE=knowledge
            drop = rag.lies_drop_types({"SOURCE": "knowledge", "DROP_TYPES": v.DROP_TYPES})
            with open(v.KNOWLEDGE_PATH) as f:
                roh = [json.loads(z) for z in f if z.strip()]
            chunks = rag.baue_knowledge_chunks(roh, drop)
            embs = rag.l2norm(rag.embed([c["text"] for c in chunks]))
            self._index, self._index_key = (chunks, embs), key
        return self._index

    def _suche(self, frage):
        with self._lock:
            rag = self._lade_rag()
            chunks, embs = self._lade_index(rag)
            hits = rag.retrieve(frage, chunks, embs, k=self.valves.TOP_K)
            return rag.baue_nachrichten(frage, hits), hits

    async def _stream(self, nachrichten):
        payload = {"model": self.valves.LLM_MODEL, "messages": nachrichten,
                   "think": self.valves.THINK, "stream": True}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            async with client.stream("POST", f"{self.valves.OLLAMA_URL}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for zeile in r.aiter_lines():
                    stueck = ollama_zeile(zeile)
                    if stueck:
                        yield stueck

    async def pipe(self, body: dict, __task__=None):
        messages = body.get("messages", [])
        if __task__:
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
