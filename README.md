# Boardgame RAG Lab

Ein minimaler, lokaler RAG-Pruefstand fuer Brettspiel-Regelfragen -- Begleitcode zur
Mayflower-Blogserie ueber lokales Retrieval-Augmented-Generation auf eigener Hardware.

Kein Framework, keine Cloud: PDF einlesen, in Haeppchen schneiden, per Embedding
durchsuchbar machen, optional mit einem Reranker nachscharfstellen und von einem
lokalen LLM (via Ollama) beantworten lassen. Gedacht zum **Verstehen und Messen**,
nicht als Produktivsystem.

> **Urheberrecht:** In diesem Repo sind **keine Regelhefte** enthalten. Lege dein
> eigenes PDF nach `pdfs/` (per `.gitignore` ausgeschlossen).

## Was drin ist

| Datei | Zweck |
|---|---|
| `rag.py` | Retrieval + optionaler Reranker + Golden-Set-Eval. Alle Stellschrauben per Umgebungsvariable. |
| `ingest.py` | Bessere Ingestion fuer Text/Tabellen: Docling (Layout) + faktentreue LLM-Verbalisierung -> `knowledge.jsonl`. |
| `render.py` | Rendert PDF-Seiten als PNG (Vorstufe fuer Vision). |
| `vision_test.py` | Schickt ein Bild + Frage an ein lokales Vision-Modell (schneller Check). |
| `vision_ingest.py` | Verbalisiert eine Grafik-/Infografik-Seite per Vision und haengt jede Karte als Chunk an `knowledge.jsonl`. |
| `golden_set.example.json` | Beispiel-Testset (Food Chain Magnate). Kopiere es nach `golden_set.json` und passe es an dein Spiel an. |

## Voraussetzungen

- [Ollama](https://ollama.com/) laeuft lokal, mit den Modellen:
  ```
  ollama pull qwen3:14b     # Antwortmodell
  ollama pull bge-m3        # Embeddings (multilingual/deutsch)
  ollama pull qwen2.5vl:7b  # Vision (fuer Grafik-/Infografik-Seiten)
  ```
- Python 3.11+.

## Einrichtung -- zwei getrennte venvs

Das ist kein Zufall: Docling und sentence-transformers stellen widerspruechliche
Ansprueche an `transformers`. In einer gemeinsamen venv bricht eines von beiden.
Also trennen (was fuer eine spaetere Microservice-Architektur ohnehin passt).

```bash
# venv 1: Serving / Retrieval / Vision
python3 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU reicht; KEIN torchvision
pip install -r requirements-serve.txt
deactivate

# venv 2: Ingestion (Docling)
python3 -m venv .venv-ingest
. .venv-ingest/bin/activate
pip install -r requirements-ingest.txt
deactivate
```

## Nutzung

```bash
# PDF in den Ordner legen
cp ~/mein-spiel.pdf pdfs/

# Eine Frage (einfache Ingestion: rohe PDF-Extraktion)
. .venv/bin/activate
python rag.py ask "Wie verdiene ich Geld?"

# Testset durchlaufen
cp golden_set.example.json golden_set.json   # dann an dein Spiel anpassen
python rag.py eval
```

### Stellschrauben (alles per env)

```bash
CHUNK_SIZE=400 python rag.py eval        # Haeppchengroesse (Zeichen)
TOP_K=8 python rag.py eval               # wie viele Haeppchen in den Kontext
RERANK=1 python rag.py eval              # Reranker an (over-retrieve -> Cross-Encoder)
CANDIDATES=20 RERANK=1 python rag.py eval # Kandidatenfeld vor dem Reranking
```

Beste Kombination in unseren Tests: `CHUNK_SIZE=400 RERANK=1`. Mehr ist nicht besser
-- `TOP_K=8` und `CANDIDATES=40` haben die Ergebnisse jeweils verschlechtert.

### Ingestion nach Inhaltstyp

Es gibt keine eine beste Methode -- es haengt davon ab, was auf der Seite steht.

**Text & echte Tabellen** (Docling + Verbalisierung):

```bash
. .venv-ingest/bin/activate
python ingest.py pdfs/mein-spiel.pdf     # erzeugt knowledge.jsonl
deactivate

. .venv/bin/activate
SOURCE=knowledge CHUNK_SIZE=400 RERANK=1 python rag.py eval
```

**Grafiken & Infografiken** (Vision) -- das, woran reine Textextraktion scheitert:

```bash
. .venv/bin/activate
python render.py pdfs/mein-spiel.pdf 6                    # Grafikseite -> seite_6.png
python vision_test.py seite_6.png "Was steht auf Karte X?" # schneller Check
python vision_ingest.py seite_6.png                       # jede Karte als Chunk -> knowledge.jsonl
SOURCE=knowledge CHUNK_SIZE=400 RERANK=1 python rag.py eval
```

## Was man dabei lernt

- Die groessten Fehler sind nicht die falschen Antworten, sondern die *ueberzeugend*
  falschen. Ein sauberer Grounding-Prompt und kleine Haeppchen bringen das Modell
  dazu, lieber "keine Angabe" zu sagen als zu raten.
- "Mehr" (mehr Haeppchen, groesseres Kandidatenfeld) macht es oft schlechter.
- Der groesste Hebel sitzt ganz vorne, beim Einlesen -- und die richtige Methode
  haengt vom Inhaltstyp ab: **Fliesstext -> rohe Extraktion reicht, echte Tabellen
  -> Docling + Verbalisierung, Grafiken -> Vision.**
- Miss die richtige Sache: ohne ein Golden Set, das alle Inhaltstypen abdeckt,
  zieht man leicht den falschen Schluss.
