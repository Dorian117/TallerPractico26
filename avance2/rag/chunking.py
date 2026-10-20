"""Chunking: de bloques por seccion a fragmentos de tamano acotado.

Principios (cada uno responde a un fallo concreto de un RAG matematico):

1. RESPETAR LA ESTRUCTURA. El lector ya entrega un bloque por (sub)seccion.
   Si un bloque cabe en CHUNK_TAMANO, es UN chunk: un ejemplo resuelto
   completo vale mas que dos mitades que el modelo debe adivinar unir.
2. JAMAS CORTAR UNA FORMULA. Una mitad de `\\begin{align*}` no es LaTeX valido
   y confunde tanto al embedding como al LLM. Los entornos \\begin..\\end, $$..$$,
   \\[..\\] y $..$ son atomos indivisibles.
3. CORTES JERARQUICOS. Si hay que partir, se corta primero entre parrafos,
   luego entre lineas y, solo en ultimo caso, entre oraciones.
4. SOLAPAMIENTO CON ATOMOS COMPLETOS. La cola que se repite en el chunk
   siguiente se arma con trozos enteros, nunca a mitad de formula.
5. SIN TROZOS DIMINUTOS. Un bloque de <CHUNK_MINIMO caracteres se fusiona
   con el siguiente: 80 caracteres casi no aportan contexto.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .readers import Bloque, DocumentoLeido

# --- atomos indivisibles ---------------------------------------------------
_RE_ATOMO = re.compile(
    r"\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\begin\{([a-zA-Z*]+)\}.*?\\end\{\1\}"
    r"|(?<!\\)\$[^$\n]+?(?<!\\)\$",        # matematica en linea $...$
    re.S,
)
_MARCA = "\x00{}\x00"
_RE_MARCA = re.compile(r"\x00(\d+)\x00")


def _enmascarar(texto: str) -> tuple[str, list[str]]:
    """Sustituye cada atomo por una marca opaca para que ningun separador
    (salto de linea, punto...) que viva DENTRO de una formula parta nada."""
    atomos: list[str] = []

    def guardar(m: re.Match) -> str:
        atomos.append(m.group(0))
        return _MARCA.format(len(atomos) - 1)

    return _RE_ATOMO.sub(guardar, texto), atomos


def _restaurar(texto: str, atomos: list[str]) -> str:
    return _RE_MARCA.sub(lambda m: atomos[int(m.group(1))], texto)


# Jerarquia de separadores: (regex de corte, separador con el que se re-une)
_SEPARADORES = [
    (re.compile(r"\n\s*\n"), "\n\n"),         # parrafos
    (re.compile(r"\n"), "\n"),                # lineas
    (re.compile(r"(?<=[.!?;:])\s+"), " "),    # oraciones
]


def _longitud_real(pieza: str, atomos: list[str]) -> int:
    return len(_restaurar(pieza, atomos))


def _partir(texto: str, atomos: list[str], tam: int, nivel: int = 0) -> list[tuple[str, str]]:
    """Devuelve piezas (texto_enmascarado, separador_previo) de <= tam cuando
    es posible. Una pieza que sigue siendo > tam tras agotar la jerarquia se
    deja entera (p. ej. una tabla o un align* gigante): mejor entera que rota."""
    if _longitud_real(texto, atomos) <= tam or nivel >= len(_SEPARADORES):
        return [(texto, "")]
    regex, sep = _SEPARADORES[nivel]
    trozos = [t for t in regex.split(texto) if t.strip()]
    if len(trozos) <= 1:
        return _partir(texto, atomos, tam, nivel + 1)
    piezas: list[tuple[str, str]] = []
    for i, t in enumerate(trozos):
        sub = _partir(t, atomos, tam, nivel + 1)
        # la primera sub-pieza hereda el separador de este nivel
        piezas.append((sub[0][0], sep if i > 0 else ""))
        piezas.extend(sub[1:])
    return piezas


def trocear_texto(texto: str, tam: int | None = None, solape: int | None = None) -> list[str]:
    tam = tam or config.CHUNK_TAMANO
    solape = config.CHUNK_SOLAPE if solape is None else solape
    if len(texto) <= tam:
        return [texto]

    enmascarado, atomos = _enmascarar(texto)
    piezas = _partir(enmascarado, atomos, tam)

    chunks: list[str] = []
    actual: list[tuple[str, str]] = []   # (pieza, separador previo)

    def unir(ps: list[tuple[str, str]]) -> str:
        return _restaurar("".join((s if i else "") + p for i, (p, s) in enumerate(ps)), atomos)

    for pieza in piezas:
        if actual and len(unir(actual + [pieza])) > tam:
            chunks.append(unir(actual))
            # cola de solape: piezas completas desde el final hasta ~solape chars
            cola: list[tuple[str, str]] = []
            for p in reversed(actual):
                if len(unir([p] + cola)) > solape:
                    break
                cola.insert(0, p)
            # una pieza de la cola, al ir primera, pierde su separador previo
            actual = cola
        actual.append(pieza)
    if actual:   # siempre contiene al menos una pieza nueva (se anade tras el solape)
        chunks.append(unir(actual))
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Clasificacion del tipo de chunk (metadato filtrable)
# ---------------------------------------------------------------------------
_TIPOS = [
    ("respuesta", re.compile(r"respuestas?|soluciones de los ejercicios", re.I)),
    ("ejercicio", re.compile(r"ejercicios? (propuestos|de identificaci)|emparejamiento|"
                             r"c[aá]lculo de integrales|problemas aplicados", re.I)),
    ("ejemplo", re.compile(r"ejemplo|ejercicio [a-z]\)|caso [a-z]\b|soluci[oó]n|"
                           r"desarrollo paso|ejercicios resueltos|an[aá]lisis de selecci|"
                           r"resumen inicial", re.I)),
    ("procedimiento", re.compile(r"algoritmo|estrategia|m[eé]todo tabular|nemot[eé]cnica|ilate|pasos", re.I)),
    ("formulario", re.compile(r"f[oó]rmula|formulario|tabla|reglas b[aá]sicas|identidades|"
                              r"razones trigonom|casos est[aá]ndar|integrales b[aá]sicas|derivadas de", re.I)),
]


def clasificar_tipo(seccion: str, texto: str) -> str:
    """Tipo segun el ENCABEZADO MAS ESPECIFICO (el ultimo de la ruta): el
    titulo de seccion es una senal fiable en apuntes de curso y no cuesta nada."""
    ultimo = seccion.split(" > ")[-1] if seccion else ""
    for tipo, patron in _TIPOS:
        if patron.search(ultimo):
            return tipo
    if re.search(r"\*\*(DEFINICI[OÓ]N|TEOREMA)", texto):
        return "definicion"
    return "teoria"


# ---------------------------------------------------------------------------
# Texto para EMBEDDING (distinto del texto que se guarda y se cita)
# ---------------------------------------------------------------------------
_VERBAL = {
    r"\\int": " integral ", r"\\sqrt": " raiz ", r"\\frac": " fraccion ",
    r"\\sin": " seno ", r"\\cos": " coseno ", r"\\tan": " tangente ",
    r"\\sec": " secante ", r"\\csc": " cosecante ", r"\\cot": " cotangente ",
    r"\\ln": " logaritmo natural ", r"\\arcsin": " arcoseno ", r"\\arctan": " arcotangente ",
    r"\\theta": " theta ", r"\\implies": " implica ", r"\\cdot": " por ",
}
_RUIDO = re.compile(
    r"\\displaystyle|\\left(?![a-zA-Z])|\\right(?![a-zA-Z])|\\,|\\;|\\!|\\quad|\\qquad|\\\\|&|"
    r"\\begin\{[^}]*\}|\\end\{[^}]*\}|\\\[|\\\]|\$"
)


def normalizar_para_embedding(texto: str, modo: str = "limpio") -> str:
    """El texto guardado conserva el LaTeX completo (lo necesita el LLM y el
    estudiante). Para EMBEBER, el ruido de maquetacion (\\displaystyle, &, \\\\)
    gasta tokens (el modelo tiene tope de 512) y diluye el significado.
      crudo  : tal cual
      limpio : sin ruido de maquetacion
      verbal : limpio + macros matematicas como palabras (\\int -> 'integral'),
               para acercar el texto al vocabulario de las preguntas."""
    if modo == "crudo":
        return texto
    t = _RUIDO.sub(" ", texto)
    if modo == "verbal":
        for macro, palabra in _VERBAL.items():
            t = re.sub(macro + r"(?![a-zA-Z])", palabra, t)
    t = t.replace("**", "").replace("*", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", t)).strip()


# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    texto: str               # lo que se guarda, se cita y se envia al LLM
    texto_embedding: str     # lo que se convierte en vector
    fuente: str              # nombre del archivo
    documento: str           # titulo legible
    seccion: str
    pagina: int
    tipo: str
    formato: str
    n_chunk: int
    excede_tokens: bool = False   # atomo indivisible que no cabe en max_seq_length


def _slug(nombre: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", nombre.lower().rsplit(".", 1)[0]).strip("-")


def _fusionar_pequenos(bloques: list[Bloque], minimo: int) -> list[Bloque]:
    """Un bloque < minimo se antepone al siguiente (con su encabezado como
    titulo en negrita) para no crear fragmentos de pocas palabras."""
    salida: list[Bloque] = []
    pendiente: list[Bloque] = []
    for b in bloques:
        if len(b.texto) < minimo:
            pendiente.append(b)
            continue
        if pendiente:
            previo = "\n\n".join(
                f"**{(p.seccion.split(' > ')[-1] or 'Introduccion')}**\n{p.texto}" for p in pendiente
            )
            b = Bloque(texto=f"{previo}\n\n{b.texto}", seccion=b.seccion, pagina=b.pagina)
            pendiente = []
        salida.append(b)
    if pendiente:   # cola de documento diminuta: se une al ultimo bloque
        previo = "\n\n".join(
            f"**{(p.seccion.split(' > ')[-1] or 'Introduccion')}**\n{p.texto}" for p in pendiente
        )
        if salida:
            u = salida[-1]
            salida[-1] = Bloque(texto=f"{u.texto}\n\n{previo}", seccion=u.seccion, pagina=u.pagina)
        else:
            salida.append(Bloque(texto=previo, seccion=pendiente[0].seccion, pagina=pendiente[0].pagina))
    return salida


def _texto_embedding(doc: DocumentoLeido, bloque: Bloque, parte: str, modo: str) -> str:
    # el encabezado ayuda al embedding: "Ejemplo 2" solo no dice de que tema es
    contexto = f"{doc.titulo} > {bloque.seccion}" if bloque.seccion else doc.titulo
    return normalizar_para_embedding(f"{contexto}\n{parte}", modo)


def _trocear_bloque_en_tokens(doc, bloque, modo, contar_tokens, prefijo, max_tokens):
    """Trocea por caracteres y, si algun trozo NO CABE en el tope de tokens del
    modelo de embeddings, reintenta el bloque con trozos un 20 % menores.

    Por que tokens y no caracteres: el texto matematico tokeniza muy denso
    (~2 caracteres por token en este corpus), asi que 1800 caracteres pueden
    ocupar 800 tokens y el modelo truncaria el final EN SILENCIO. Midiendo en
    tokens, un ejemplo resuelto queda entero cuando cabe y se parte solo
    cuando de verdad no cabe (regla "entero si cabe")."""
    tam = config.CHUNK_TAMANO
    while True:
        partes = trocear_texto(bloque.texto, tam)
        if contar_tokens is None:
            return partes, []
        exceden = [
            p for p in partes
            if contar_tokens(prefijo + _texto_embedding(doc, bloque, p, modo)) > max_tokens
        ]
        # tam < 300: ya no se puede partir mas sin romper atomos (formula o
        # tabla indivisible mayor que el tope); se acepta y se reporta.
        if not exceden or tam < 300:
            return partes, exceden
        tam = int(tam * 0.8)


def trocear_documento(
    doc: DocumentoLeido,
    modo_embedding: str = "limpio",
    contar_tokens=None,
    prefijo: str = "",
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Si se pasa `contar_tokens`, ningun chunk (con su prefijo) supera
    `max_tokens` salvo atomos indivisibles; esos quedan en `Chunk.excede_tokens`."""
    max_tokens = max_tokens or config.EMBEDDING_MAX_TOKENS
    chunks: list[Chunk] = []
    base = _slug(doc.archivo)
    for bloque in _fusionar_pequenos(doc.bloques, config.CHUNK_MINIMO):
        partes, exceden = _trocear_bloque_en_tokens(
            doc, bloque, modo_embedding, contar_tokens, prefijo, max_tokens)
        for parte in partes:
            n = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{base}-{n:03d}",
                texto=parte,
                texto_embedding=_texto_embedding(doc, bloque, parte, modo_embedding),
                fuente=doc.archivo,
                documento=doc.titulo,
                seccion=bloque.seccion,
                pagina=bloque.pagina,
                tipo=clasificar_tipo(bloque.seccion, parte),
                formato=doc.formato,
                n_chunk=n,
                excede_tokens=parte in exceden,
            ))
    return chunks
