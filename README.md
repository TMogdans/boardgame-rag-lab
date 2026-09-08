# Boardgame RAG Lab

Ein minimaler, lokaler RAG-Pruefstand fuer Brettspiel-Regelfragen -- Begleitcode zur
Mayflower-Blogserie ueber lokales Retrieval-Augmented-Generation auf eigener Hardware.

Kein Framework, keine Cloud: PDF einlesen, in Haeppchen schneiden, per Embedding
durchsuchbar machen, optional mit einem Reranker nachscharfstellen und von einem
lokalen LLM (via Ollama) beantworten lassen. Gedacht zum **Verstehen und Messen**,
nicht als Produktivsystem.

> **Urheberrecht:** In diesem Repo sind **keine Regelhefte** enthalten. Lege dein
> eigenes PDF nach `pdfs/` (per `.gitignore` ausgeschlossen).

## Schnellstart

```bash
git clone git@github.com:TMogdans/boardgame-rag-lab.git
cd boardgame-rag-lab

# 1. Ollama + Modelle bereitstellen (Details unter "Voraussetzungen")
# 2. Serving-venv einrichten
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU reicht; KEIN torchvision
pip install -r requirements-serve.txt

# 3. eigenes Regel-PDF ablegen und die erste Frage stellen
cp ~/mein-spiel.pdf pdfs/
python rag.py ask "Wie verdiene ich Geld?"
```

Das ist der schnellste Weg zur ersten Antwort (rohe PDF-Extraktion, ein Modell,
eine venv). Bessere Ingestion fuer Tabellen und Grafiken sowie die zweite venv
sind weiter unten beschrieben.

## Was drin ist

| Datei | Zweck |
|---|---|
| `rag.py` | Retrieval + optionaler Reranker + Golden-Set-Eval. Alle Stellschrauben per Umgebungsvariable. |
| `ingest.py` | Bessere Ingestion fuer Text/Tabellen: Docling (Layout) + faktentreue LLM-Verbalisierung -> `knowledge.jsonl`. |
| `render.py` | Rendert PDF-Seiten als PNG (Vorstufe fuer Vision). |
| `vision_test.py` | Schickt ein Bild + Frage an ein lokales Vision-Modell (schneller Check). |
| `vision_ingest.py` | Verbalisiert eine Grafik-/Infografik-Seite per Vision und haengt jede Karte als Chunk an `knowledge.jsonl`. |
| `auto_ingest.py` | **Auto-Router:** entscheidet pro Seite selbst zwischen Text/Tabelle/Vision (Docling-Layout + Fragment-Heuristik) -> `knowledge.jsonl`. |
| `classify.py` | Taggt jeden Chunk als `regel`/`flavor`/`meta`, damit sich Ballast beim Retrieval ausfiltern laesst. |
| `inspect_layout.py` | Zeigt die Docling-Region-Labels pro Seite (zum Debuggen und Verstehen des Routings). |
| `golden_set.example.json` | Beispiel-Testset (Food Chain Magnate). Kopiere es nach `golden_set.json` und passe es an dein Spiel an. |

## Voraussetzungen

- [Ollama](https://ollama.com/) laeuft lokal, mit den Modellen:
  ```
  ollama pull qwen3:14b     # Antwortmodell
  ollama pull bge-m3        # Embeddings (multilingual/deutsch)
  ollama pull qwen2.5vl:7b  # Vision (fuer Grafik-/Infografik-Seiten)
  ```
  Ollama selbst startet je nach Plattform unterschiedlich. Auf einer AMD-GPU unter
  Linux laeuft es z.B. als ROCm-Container:
  ```bash
  podman run -d --device /dev/kfd --device /dev/dri \
    -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 \
    --name ollama docker.io/ollama/ollama:rocm
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
CHUNK_SIZE=400 python rag.py eval         # Haeppchengroesse (Zeichen), Default 800
CHUNK_OVERLAP=150 python rag.py eval      # Ueberlappung, Default 150 -- siehe Warnung unten
TOP_K=8 python rag.py eval                # wie viele Haeppchen in den Kontext, Default 4
RERANK=1 python rag.py eval               # Reranker an (over-retrieve -> Cross-Encoder)
CANDIDATES=40 RERANK=1 python rag.py eval # Kandidatenfeld vor dem Reranking, Default 20
SOURCE=knowledge python rag.py eval       # knowledge.jsonl statt roher PDF-Extraktion
DROP_TYPES=flavor,meta python rag.py eval # Ballast aus dem Index werfen (nur mit SOURCE=knowledge)
THINK=1 python rag.py eval                # Reasoning des LLM anschalten
```

`FRAGMENT_THRESHOLD=50` (in `auto_ingest.py`) entscheidet, ab wie vielen Text-Fragmenten
eine Seite als Grafik gilt und zur Vision-Route geht.

> **`CHUNK_OVERLAP` skaliert nicht mit `CHUNK_SIZE`.** Die Ueberlappung ist absolut, nicht
> relativ. Wer `CHUNK_SIZE` von 800 auf 400 halbiert, aendert damit **vier** Dinge auf
> einmal: Haeppchengroesse, relative Ueberlappung (18,8 % -> 37,5 %), Anzahl der Chunks
> im Index und -- bei festem `TOP_K` -- die Menge Kontext, die beim Modell ankommt
> (4 x 800 -> 4 x 400 Zeichen). Ein Ergebnisunterschied laesst sich deshalb **nicht**
> allein der Haeppchengroesse zuschreiben. Wer die Groesse isoliert messen will, muss
> `CHUNK_OVERLAP` mitskalieren und `TOP_K` gegenrechnen.

> **Ergebniszahlen aus frueheren Laeufen sind nicht mehr vergleichbar.** Die Wertung in
> `rag.py eval` wurde repariert (siehe "Was die Eval misst"): fruehere Quoten hatten einen
> Nenner, den die Lauf-Konfiguration mitverschieben konnte. Alle Vergleiche muessen mit
> diesem Stand neu erhoben werden.

### Was die Eval misst -- und was nicht

`python rag.py eval` erhebt **zwei** Dinge, und beide sind Regressionswarner, kein
Korrektheitsmass:

1. **Retrieval-Dreiteilung** `getroffen / verfehlt`, plus `Verweigerungsfragen` als eigene
   Kategorie. Ob eine Frage in die Retrieval-Wertung eingeht, haengt allein an der Frage
   selbst (Feld `erwartet_verweigerung` im Golden Set), **nie** an der Konfiguration des
   Laufs. Der Nenner ist damit ueber alle `DROP_TYPES`-Varianten konstant -- eine
   Verbesserung kann nicht mehr dadurch entstehen, dass Fragen aus der Wertung fallen.
2. **Keyword-Treffer** in der Antwort, auf Wortgrenzen. Das ist ein grober
   Aenderungsdetektor: das Skript kennt **keine** Kategorie "falsch" und kann
   "keine einzige falsche Antwort" nicht ermitteln. Wer eine Korrektheitsaussage braucht,
   liest die Antworten selbst gegen das Heft.

Das Golden Set kennzeichnet Verweigerungsfragen ausdruecklich:

```json
{ "frage": "Gibt es die Spielregel 'Gleicher Mist, doppelter Preis'?",
  "typ": "flavor", "erwartet_verweigerung": true, "seiten": [2] }
```

`typ` ist eine inhaltliche Kategorie und **keine** Wertungsanweisung -- eine Frage ohne
`seiten` und ohne `erwartet_verweigerung` ist ein Konfigurationsfehler und bricht ab.

**Die Eval bricht ab statt stillschweigend etwas anderes zu messen**, wenn:

- `DROP_TYPES` einen Wert nennt, den `classify.py` nicht vergibt (`regel`/`flavor`/`meta`) --
  insbesondere bei Verwechslung mit den Frage-Typen des Golden Sets (`fakt`, `falle`,
  `leerstelle`, `tabelle`);
- `DROP_TYPES` ohne `SOURCE=knowledge` gesetzt ist (der Filter koennte dort nicht wirken);
- `DROP_TYPES` gesetzt ist, aber kein Chunk ein `typ`-Feld hat (`classify.py` fehlt);
- ein `knowledge.jsonl`-Eintrag kein Feld `seite` hat (fruehere Faelle lieferten die
  Chunk-ID als Seitenzahl aus -- eine erfundene Fundstelle ist schlimmer als ein Abbruch);
- `CHUNK_SIZE <= CHUNK_OVERLAP` (waere eine Endlosschleife).

> **Alte `knowledge.jsonl` neu erzeugen.** Dateien, die vor diesem Stand geschrieben wurden,
> haben kein `seite`-Feld und fuehren zum Abbruch. Die Ingestion muss einmal neu laufen --
> es genuegt nicht, nur `rag.py` zu aktualisieren.

### Tests

Die Auswertungslogik ist ohne Ollama, ohne Modelle und ohne PDF pruefbar:

```bash
python test_wertung.py       # Dreiteilung, DROP_TYPES-Entkopplung, Keywords, Guards
python test_classify.py      # Klassifikator-Auswertung, 17 Antwortvarianten
python test_ingest_seite.py  # 'seite'-Feld in ingest.py und vision_ingest.py
python test_mutationen.py    # Mutationsprobe: verfaelscht die Fixes und prueft, dass Tests rot werden
```

### Ingestion nach Inhaltstyp

Es gibt keine eine beste Methode -- es haengt davon ab, was auf der Seite steht.

**Alles automatisch** (Auto-Router -- der bequemste Weg): entscheidet pro Seite selbst,
welcher der folgenden Wege genommen wird.

```bash
. .venv-ingest/bin/activate
python auto_ingest.py pdfs/mein-spiel.pdf   # routet jede Seite selbst -> knowledge.jsonl
deactivate

. .venv/bin/activate
python classify.py                          # optional: Ballast als flavor/meta taggen
                                            # schreibt die rohe Modellantwort als 'typ_antwort' mit,
                                            # damit eine Fehl-Einordnung nachvollziehbar bleibt
SOURCE=knowledge CHUNK_SIZE=400 RERANK=1 DROP_TYPES=flavor,meta python rag.py eval
```

Wer die einzelnen Wege lieber von Hand steuert:

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
                                                          # Seite kommt aus dem Dateinamen (seite_N.png);
                                                          # bei anderem Namen: SEITE=6 davorsetzen
SOURCE=knowledge CHUNK_SIZE=400 RERANK=1 python rag.py eval
```

## Was man dabei lernt

- Die groessten Fehler sind nicht die falschen Antworten, sondern die *ueberzeugend*
  falschen. Ein sauberer Grounding-Prompt und kleine Haeppchen bringen das Modell
  dazu, lieber "keine Angabe" zu sagen als zu raten.
- "Mehr" (mehr Haeppchen, groesseres Kandidatenfeld) macht es oft schlechter.
- Der Beobachtungspunkt darf nicht an derselben Konfiguration haengen, die er bewachen soll.
  Genau das war hier eine Zeit lang der Fall: `DROP_TYPES` verkleinerte den Nenner der
  Retrieval-Quote, ohne dass ein Retrieval besser wurde.
- Der groesste Hebel sitzt ganz vorne, beim Einlesen -- und die richtige Methode
  haengt vom Inhaltstyp ab: **Fliesstext -> rohe Extraktion reicht, echte Tabellen
  -> Docling + Verbalisierung, Grafiken -> Vision.**
- Miss die richtige Sache: ohne ein Golden Set, das alle Inhaltstypen abdeckt,
  zieht man leicht den falschen Schluss.
