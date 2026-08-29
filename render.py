#!/usr/bin/env python3
"""
Rendert einzelne PDF-Seiten als PNG -- Vorstufe fuer den Vision-Schritt.

    python render.py pdfs/mein-spiel.pdf 5 6      # rendert Seite 5 und 6
    python render.py 5 6                           # nimmt das erste PDF in pdfs/

Aufloesung ueber die Umgebungsvariable DPI (Default 150).
"""
import sys, os, glob
import pymupdf

BASE = os.path.dirname(os.path.abspath(__file__))
DPI  = int(os.environ.get("DPI", 150))

args   = sys.argv[1:]
pdfs   = [a for a in args if a.lower().endswith(".pdf")]
seiten = [int(a) for a in args if a.isdigit()]

pdf = pdfs[0] if pdfs else next(iter(sorted(glob.glob(os.path.join(BASE, "pdfs", "*.pdf")))), None)
if not pdf:
    sys.exit("Kein PDF gefunden. Nutzung: python render.py <pdf> <seite...>")
if not seiten:
    sys.exit("Keine Seiten angegeben. Nutzung: python render.py <pdf> 5 6")

doc = pymupdf.open(pdf)
for s in seiten:
    pix = doc[s - 1].get_pixmap(dpi=DPI)  # pymupdf ist 0-basiert
    out = os.path.join(BASE, f"seite_{s}.png")
    pix.save(out)
    print(f"{out}  ({pix.width}x{pix.height})")
