"""Prueba de humo de lectores y chunker (sin red, sin API).

    python -m tests.smoke_readers

El corpus real solo trae .txt con LaTeX, asi que PDF, DOCX, HTML y MD se
prueban aqui con archivos SINTETICOS minimos. Esto demuestra que cada lector
funciona sobre un caso sencillo; NO demuestra calidad sobre documentos reales
de otro origen (para PDF con formulas, ver docs/DECISIONES.md, seccion Ingesta).
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402
from rag.chunking import trocear_texto  # noqa: E402
from rag.readers import leer_documento  # noqa: E402


def probar_lectores(d: Path) -> None:
    (d / "a.md").write_text("# Titulo A\nintro\n## Seccion 1\ntexto uno\n```\n# no es encabezado\n```\n## Seccion 2\ntexto dos\n", encoding="utf-8")
    (d / "b.txt").write_text("Texto plano sin latex.", encoding="utf-8")
    (d / "c.txt").write_text(r"\documentclass{article}\begin{document}\section{Hola}Texto con $x^2$.\end{document}", encoding="utf-8")
    (d / "d.html").write_text("<html><head><title>Doc HTML</title><script>var x=1;</script></head><body>"
                              "<h1>Uno</h1><p>parrafo uno</p><h2>Dos</h2><p>parrafo dos</p></body></html>", encoding="utf-8")

    import docx
    dx = docx.Document()
    dx.add_heading("Encabezado Word", level=1)
    dx.add_paragraph("cuerpo del docx")
    dx.add_table(rows=1, cols=2).rows[0].cells[0].text = "celda"
    dx.save(d / "e.docx")

    import pymupdf
    pdf = pymupdf.open()
    p = pdf.new_page()
    p.insert_text((72, 100), "Titulo grande", fontsize=24)
    p.insert_text((72, 150), "Cuerpo del pdf con texto normal de prueba.", fontsize=11)
    p.insert_text((72, 170), "Segunda linea del cuerpo del pdf.", fontsize=11)
    pdf.save(d / "f.pdf")
    pdf.close()

    md = leer_documento(d / "a.md")
    assert [b.seccion for b in md.bloques] == ["Titulo A", "Titulo A > Seccion 1", "Titulo A > Seccion 2"], md.bloques
    assert "# no es encabezado" in md.bloques[1].texto           # '#' dentro de ``` no abre seccion

    assert leer_documento(d / "b.txt").formato == "txt"
    tx = leer_documento(d / "c.txt")
    assert tx.formato == "tex" and tx.bloques[0].seccion == "Hola" and "$x^2$" in tx.bloques[0].texto

    ht = leer_documento(d / "d.html")
    assert ht.titulo == "Doc HTML" and [b.seccion for b in ht.bloques] == ["Uno", "Uno > Dos"], ht.bloques
    assert "var x" not in "".join(b.texto for b in ht.bloques)      # <script> descartado

    dc = leer_documento(d / "e.docx")
    assert dc.bloques[0].seccion == "Encabezado Word" and "cuerpo del docx" in dc.bloques[0].texto and "celda" in dc.bloques[0].texto

    pf = leer_documento(d / "f.pdf")
    assert pf.bloques[0].pagina == 1 and pf.bloques[0].seccion == "Titulo grande", pf.bloques
    assert "Cuerpo del pdf" in pf.bloques[0].texto and "Titulo grande" not in pf.bloques[0].texto
    print("lectores OK: md, txt, txt-con-latex, html, docx, pdf (sinteticos)")


def probar_chunker() -> None:
    # texto largo con muchas formulas protegidas: ningun chunk puede cortarlas
    bloque_mate = "\\begin{align*}\n a &= b + c \\\\\n d &= e + f\n\\end{align*}"
    texto = "\n\n".join(
        f"Parrafo {i}. Evaluar la integral $\\int x^{i}\\,dx$ y comprobar. " + bloque_mate + f"\n\nDisplay: \\[ y_{i} = {i} \\] y $$ z_{i} $$."
        for i in range(30)
    )
    for tam, solape in [(300, 60), (700, 150), (1800, 200)]:
        chunks = trocear_texto(texto, tam, solape)
        assert len(chunks) > 1
        for c in chunks:
            assert len(re.findall(r"\\begin\{align\*\}", c)) == len(re.findall(r"\\end\{align\*\}", c)), "entorno cortado"
            assert c.count("$$") % 2 == 0, "$$ cortado"
            assert len(re.findall(r"\\\[", c)) == len(re.findall(r"\\\]", c)), r"\[ \] cortado"
            assert len(re.findall(r"(?<!\\)\$", c.replace("$$", ""))) % 2 == 0, "$ en linea cortado"
        # nada se pierde: cada parrafo original aparece entero en algun chunk
        for i in range(30):
            assert any(f"Parrafo {i}." in c for c in chunks)
    # un texto corto no se parte
    assert trocear_texto("corto", 100, 10) == ["corto"]
    print("chunker OK: ningun chunk corta align*, $$, \\[ \\] ni $...$ (tam 300/700/1800)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        probar_lectores(Path(t))
    probar_chunker()
    print("TODO OK")
