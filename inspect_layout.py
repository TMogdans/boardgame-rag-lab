#!/usr/bin/env python3
"""Zeigt, welche Regionstypen Docling pro Seite erkennt -- Grundlage fuer den Auto-Router.

Laeuft in der Ingest-venv (docling).
"""
import os
from collections import Counter
from docling.document_converter import DocumentConverter

BASE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(BASE, "pdfs", "FCM_Rules_DE_v3.pdf")

doc = DocumentConverter().convert(PDF).document

per_page = {}
for item, _level in doc.iterate_items():
    prov = getattr(item, "prov", None)
    page = prov[0].page_no if prov else "?"
    label = getattr(item, "label", None) or type(item).__name__
    per_page.setdefault(page, Counter())[str(label)] += 1

for page in sorted(per_page, key=lambda p: (p == "?", p)):
    counts = per_page[page]
    print(f"Seite {page:>2}: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
