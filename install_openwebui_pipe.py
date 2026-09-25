#!/usr/bin/env python3
"""
Laedt openwebui_pipe.py als Funktion in Open WebUI (anlegen oder aktualisieren) und schaltet sie ein.

    OPENWEBUI_URL=http://drachenhort.local:8080 OPENWEBUI_KEY=sk-... python install_openwebui_pipe.py

Braucht einen Admin-API-Key. Die Datei im Repo bleibt die Quelle: nach jeder
Aenderung an openwebui_pipe.py dieses Skript erneut laufen lassen. Aenderungen an
rag.py brauchen das nicht -- die Pipe laedt rag.py selbst aus dem gemounteten Clone.
"""
import os, sys
import requests

FUNKTION_ID = "brettspiel_rag"
NAME = "Brettspiel-Regeln (RAG)"
BESCHREIBUNG = "Regelfragen aus knowledge.jsonl, gleiche Retrieval-Logik wie rag.py"


def main():
    url = os.environ.get("OPENWEBUI_URL", "http://127.0.0.1:8080").rstrip("/")
    key = os.environ.get("OPENWEBUI_KEY")
    if not key:
        sys.exit("OPENWEBUI_KEY fehlt (Admin-API-Key aus Open WebUI).")
    h = {"Authorization": f"Bearer {key}"}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "openwebui_pipe.py")) as f:
        form = {"id": FUNKTION_ID, "name": NAME, "content": f.read(),
                "meta": {"description": BESCHREIBUNG}}

    vorhanden = requests.get(f"{url}/api/v1/functions/id/{FUNKTION_ID}", headers=h, timeout=30)
    if vorhanden.status_code == 200:
        r = requests.post(f"{url}/api/v1/functions/id/{FUNKTION_ID}/update", headers=h, json=form, timeout=60)
        aktion = "aktualisiert"
    else:
        r = requests.post(f"{url}/api/v1/functions/create", headers=h, json=form, timeout=60)
        aktion = "angelegt"
    r.raise_for_status()

    aktiv = r.json().get("is_active")
    if not aktiv:
        t = requests.post(f"{url}/api/v1/functions/id/{FUNKTION_ID}/toggle", headers=h, timeout=30)
        t.raise_for_status()
        aktiv = t.json().get("is_active")
    print(f"Funktion {FUNKTION_ID} {aktion}, aktiv={aktiv}")


if __name__ == "__main__":
    main()
