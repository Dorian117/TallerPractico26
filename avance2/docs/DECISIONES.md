# Decisiones técnicas — Tutor de Cálculo II con RAG (Avance 2)

Documento para alguien que está **aprendiendo RAG**: por cada componente, qué se
decidió, qué alternativas había, por qué, y con qué parámetros finales. Todas las
cifras salen de ejecuciones reales (comando indicado). Estado: **Fase 1 completa**
(demo conversacional incluida, sección 7); Fase 2 con la infraestructura de
evaluación lista pero sin dataset ni resultados; Fase 3 pendiente.

> Un RAG (*Retrieval-Augmented Generation*) responde en dos tiempos: **recuperar**
> los fragmentos del material que mejor se parecen a la pregunta y **generar** la
> respuesta con un LLM al que se le entregan solo esos fragmentos. La calidad del
> conjunto la limita el eslabón más débil: si la recuperación trae el fragmento
> equivocado, el mejor LLM no puede arreglarlo.

---

## 0. Entorno

| Decisión | Python **3.12.10**, venv en `avance2/.venv` |
|---|---|
| Alternativa | Python 3.14 (el que había instalado) |
| Por qué | Con 3.14, `scikit-network` (dependencia de Ragas) **no tiene wheel**: pip intentaba compilarlo y falló (`Microsoft Visual C++ 14.0 or greater is required`). Con `pip install --dry-run --only-binary=:all:` se comprobó que 3.12 y 3.13 resuelven **todo el conjunto solo con wheels, en Windows y Linux**. Se eligió 3.12 por ser el mínimo que exige numpy 2.5.3 y el más maduro del ecosistema. |

Dos versiones fijadas a mano por problemas **comprobados**, no por gusto
(`requirements-eval.txt`):

- `instructor==1.17.0`: sin fijarla, el resolvedor de pip retrocedía hasta la 1.3.2 (muy antigua).
- `langchain-community==0.4.1`: Ragas 0.4.3 hace `from langchain_community.chat_models.vertexai import ChatVertexAI`, módulo que la 0.4.2 eliminó (comprobado inspeccionando los wheels: está en 0.4.0 y 0.4.1, no en 0.4.2).

Lección: un `--dry-run` resuelve versiones pero **no compila**; no detecta wheels ausentes.

Separación de dependencias: `requirements.txt` (ejecución, 10 paquetes; lo que
instalará el Space) y `requirements-eval.txt` (añade Ragas, ~43 paquetes más:
langchain, openai, datasets…). Así el despliegue no carga con lo que solo hace
falta para evaluar.

---

## 1. Corpus

**Qué hay:** 6 apuntes del curso de Cálculo II (docente: Dr. Óscar Leonardo
Pantano-Mogollón), entregados como `.txt` que contienen **código fuente LaTeX**
(~66 K caracteres; 49 888 tras la limpieza). Son el "documento fuente editable"
de cada tema, que es lo preferible a un PDF.

| Archivo | Título real (en el documento) | Pertinencia al dominio |
|---|---|---|
| `…antiderivaci_n.txt` | Cálculo II: Antiderivación | Base: antiderivada, reglas básicas, PVI, cinemática |
| `…simplificaci_n_algebraica.txt` | Cálculo II: Simplificación Algebraica | Técnica 4 del dominio (reescribir antes de integrar) |
| `…sustitucion.txt` | MÉTODO DE INTEGRACIÓN: SUSTITUCIÓN | Cambio de variable (prerrequisito de la combinación sustitución+partes) |
| `…integraci_n_por_partes.txt` | Cálculo II: Método de Integración por Partes | Partes, ILATE, método tabular, ejemplo combinado con sustitución |
| `…Areas_entre-Curvas.txt` | **INTEGRALES TRIGONOMÉTRICAS** | Potencias de sen/cos/tan/sec; ángulo doble/medio |
| `…susticion_trigonometrica.txt` | SUSTITUCIÓN TRIGONOMÉTRICA | Técnica 1 del dominio |

**Hallazgo 1 — el nombre del archivo no siempre describe el contenido.**
`…Areas_entre-Curvas.txt` trata de **integrales trigonométricas**, no de áreas.
Por eso cada fragmento lleva el título del documento (`documento=`) además del
nombre de archivo (`fuente=`), y las citas usan el título.

**Hallazgo 2 — errores matemáticos en las tablas de respuestas.** Se
auditaron **todas** las respuestas y ejemplos de los 6 apuntes **derivando
numéricamente** (razón `F'(x)/f(x)`; vale 1 si la antiderivada es correcta; 68
comprobaciones, script en el historial de la sesión). Resultado: 6 respuestas
erróneas, 62 correctas. Antes de corregir se habían revisado algunas solo a mano, y
esa revisión falló una vez (dio por bueno un resultado que la derivación refutó).

**Corregidas (con tu autorización) en `corpus/`:**

| Dónde | Decía | `F'/f` | Ahora (`F'/f = 1`) |
|---|---|---|---|
| Simplificación, ej. a) (resumen y paso 6) | `30√t + 20√(t³) − 6√(t⁵) + C` | **15** | `(30√t + 20√(t³) − 6√(t⁵))/15 + C` |
| Simplificación, propuesto h) | `… + t³ + 3t² + C` | no constante | `… + t³ + (3/2)t² + C` |
| Sust. trigonométrica, propuesto b) | `−(1/3) ln|(√(4x²+9) − 3)/(2x)|` | **−1** | `+(1/3) ln|…|` |
| Sust. trigonométrica, propuesto e) | `4(2/√((x−1)²−4)) + arctan(2/√((x−1)²−4))` | no constante | `√((x−1)²−4) − 2 sec⁻¹((x−1)/2) + C` |

**SIN corregir, pendientes de tu decisión** (se encontraron después; en el apunte
de *integración por partes*, ejercicios propuestos):

| Ítem | Problema |
|---|---|
| e) `∫ x²/√(36−x²) dx` | La respuesta del documento `−√(36−x²)` es la de `∫ x/√(36−x²) dx` (`F'/f` no constante: 0,83 / 0,40 / 0,25). O el enunciado debía ser `x/√(36−x²)`, o la respuesta correcta es `18 arcsen(x/6) − (x/2)√(36−x²) + C`. Hay que elegir. |
| i) `∫ x⁻⁵ ln x dx` | La respuesta `−ln x/(4x⁴) − 1/(8x⁴)` debe terminar en `− 1/(16x⁴)` (con `1/16`, `F'/f = 1`). |

Estas dos **no se usan** como `ground_truth`.

**Hallazgo 3 — `…Areas_entre-Curvas.txt` está truncado.** Termina a mitad de un
título de sección (`\section{Integrales de Productos de Senos y Cosenos con
Argumentos Dist`) y sin `\end{document}`: se cortó al copiarlo. El contenido
anterior está completo; falta el final del documento (que parece tratar productos
de senos y cosenos con argumentos distintos).

**Alcance.** El dominio pedido incluía "integrales trigonométricas con ángulo
doble/medio" y "sustitución + partes"; el corpus los cubre. Quedan **fuera del
corpus** (comprobado buscando el texto): sumas de Riemann, áreas entre curvas,
volúmenes/discos, ecuaciones diferenciales, integral definida. Las fracciones
parciales aparecen una sola vez, como etiqueta ("no cumple directamente,
fracciones parciales"), sin explicar el método.

*Historia:* se probó primero con 16 PDF de las mismas "Étud" (exportados de Word,
fuente CambriaMath). La extracción de fórmulas fue **mala con `pypdf` y con
`pymupdf`**: la integral salía como `Q` o `g`, los exponentes como `!` o `-`
(`5𝑥- + 20𝑥+ 6` por `5x²+20x+6`), las raíces como `M/b/e`, y el mapeo cambia de un
PDF a otro, así que no hay reparación universal. Para un tutor que debe vigilar
errores de signo eso es inaceptable; de ahí el cambio a fuentes LaTeX.

---

## 2. Ingesta (`rag/readers.py`, `rag/ingest.py`)

Un lector por formato, todos devuelven `DocumentoLeido` = lista de `Bloque`
(texto + sección + página):

| Formato | Lector | Sección (metadato) | Probado con |
|---|---|---|---|
| `.tex`, y `.txt` que contiene LaTeX (se detecta por contenido) | propio, sin librerías | pila de `\section/\subsection/\subsubsection` | **corpus real** |
| `.md` | propio | pila de `#` (ignora `#` dentro de ```) | archivo sintético |
| `.html` | BeautifulSoup | `<h1>`–`<h6>` | archivo sintético |
| `.docx` | python-docx | estilos "Heading n" | archivo sintético |
| `.pdf` | PyMuPDF | línea corta con letra ≥1,2× la mediana | archivo sintético |

> Honestidad: PDF/DOCX/HTML/MD están probados solo con archivos mínimos
> generados por `tests/smoke_readers.py`, porque el corpus real no los contiene.
> El lector de PDF añade **advertencias de calidad** (líneas de ≤2 caracteres,
> caracteres de área privada…) precisamente por lo que ocurrió con los PDF de
> Word. El de DOCX avisa si hay ecuaciones OMML (python-docx las ignora; se
> rescata su texto lineal).

**Limpieza del LaTeX (decisión central).** El LLM y el estudiante necesitan ver
las fórmulas *en LaTeX*, pero no el ruido de maquetación. Se conserva intacta
toda la matemática (`align*`, `equation*`, `$…$`, `\[…\]`) y se eliminan
preámbulo, comentarios, `\vspace`, `\hline`, el membrete institucional y las
líneas `---` (separadores que no son LaTeX válido). Las tablas pasan a filas
`a | b | c` (conservando las fórmulas de las celdas), las listas a `- `, y del
diagrama TikZ se rescata el **texto de los nodos** (los 4 pasos del algoritmo de
simplificación) en lugar de perderlo. Limitación conocida: las listas anidadas
quedan aplanadas.

**Reproducibilidad.** `python -m rag.ingest` reconstruye todo el índice (borra y
recrea la colección): ejecutado 3 veces dio siempre 88 chunks, 88 ids únicos y la
misma huella. Deduplica por SHA-256 (un archivo idéntico duplicaría cada
fragmento en el top_k) y no se detiene por un archivo ilegible: lo reporta.

---

## 3. Chunking (`rag/chunking.py`, parámetros en `rag/config.py`)

**Parámetros finales:** `CHUNK_TAMANO=1800` (techo en caracteres),
`CHUNK_SOLAPE=200`, `CHUNK_MINIMO=200`, y **límite real en tokens = 512**
(`EMBEDDING_MAX_TOKENS`).

**Cómo se llegó (el punto de partida pedido era ~1000/150–200, "valídalo con
datos"):** se midió sobre el corpus real. Datos de partida: 82 secciones con
texto; mediana 550 caracteres, p90 1 292, máximo 1 934. De 37 secciones que son
ejemplos resueltos:

| Estrategia | chunks | ejemplos partidos | chunks que superan 512 tokens |
|---|---|---|---|
| Solo caracteres 1000/150 | 87 | 6 de 37 | 3 |
| Solo caracteres 1800/200 | 76 | 0 de 37 | **9** |
| **Por tokens (tope 512), techo 1800/200** | **88** | **6 de 37** | **1** (átomo indivisible) |

El hallazgo clave: el texto matemático tokeniza muy denso (**~2,1 caracteres por
token**), así que 1 800 caracteres pueden ser 800 tokens, y el modelo de
embeddings **trunca en silencio** lo que excede 512. Mantener "el ejemplo entero
si cabe" y "no superar `max_seq_length`" solo es coherente si **"cabe" se mide en
tokens**. Implementación: se trocea por caracteres y, si algún trozo (con su
prefijo `passage: ` y su encabezado) supera 512 tokens, se reintenta el bloque
con trozos un 20 % menores. Resultado final: 88 chunks; caracteres
mín/mediana/máx = 103/543/1 322; tokens 68/265/654.

Reglas del chunker (cada una evita un fallo concreto):

1. **Un bloque por sección:** si cabe, es un chunk (un ejemplo resuelto completo vale más que dos mitades).
2. **Átomos indivisibles:** `\begin{…}\end{…}`, `$$…$$`, `\[…\]` y `$…$` nunca se cortan (el invariante se comprueba en `tests/smoke_readers.py` con tamaños 300/700/1800).
3. **Cortes jerárquicos:** párrafos → líneas → oraciones.
4. **Solapamiento con piezas completas** (nunca a mitad de fórmula).
5. **Sin trozos diminutos:** un bloque < 200 caracteres se antepone al siguiente (con su título en negrita).

**Limitación abierta:** 1 chunk (`Ejemplo 1: Potencias pares de coseno`, 1 322
caracteres, un `align*` indivisible) supera 512 tokens: su embedding solo "ve" el
inicio; el texto completo sí se guarda y se envía al LLM. El tamaño de chunk es
candidato a la iteración de la Fase 2.5.

---

## 4. Embeddings (`rag/embeddings.py`)

Modelos locales con `sentence-transformers` (nada sale de la máquina).

| | `intfloat/multilingual-e5-small` | `paraphrase-multilingual-MiniLM-L12-v2` |
|---|---|---|
| Parámetros / dimensión | 117 654 272 / 384 | 117 654 272 / 384 |
| `max_seq_length` | **512** | **128** |
| Peso en disco | ~471 MB | ~470 MB |
| Prefijos | `query: ` / `passage: ` | no |
| Licencia | MIT | Apache-2.0 |
| Entrenado para | recuperación (consulta↔pasaje) | similitud entre paráfrasis |

Pesan lo mismo (misma arquitectura): la diferencia no es el coste de CPU sino el
**límite de contexto**. Con 128 tokens el corpus se parte en 363 chunks (4×).
Benchmark real (`python -m eval.bench_embeddings`: 15 preguntas del corpus con la
sección esperada + 5 fuera; AUC = probabilidad de que una pregunta del corpus
tenga mejor top-1 que una de fuera):

| Configuración | chunks | hit@1 | hit@4 | MRR | AUC |
|---|---|---|---|---|---|
| e5 / crudo | 96 | 0,80 | 0,80 | 0,80 | 0,99 |
| **e5 / limpio** | 88 | 0,80 | **0,93** | 0,84 | 1,00 |
| e5 / verbal | 84 | 0,87 | 0,93 | 0,89 | 1,00 |
| MiniLM / limpio | 363 | 0,53 | 0,67 | 0,57 | 0,87 |
| MiniLM / verbal | 359 | 0,53 | 0,53 | 0,53 | 0,88 |

**Elegido: E5-small + texto "limpio".** E5 gana con claridad. Entre `limpio` y
`verbal` (que traduce `\int`→"integral", `\sqrt`→"raíz"…) la diferencia es **una
pregunta de 15: no concluyente**; se queda el más simple y `verbal` es candidato
para la Fase 2.5. Normalizar el texto *que se embebe* (sin tocar el que se
guarda) sí importa: con LaTeX crudo se parten 12 de 37 ejemplos frente a 6.
Además se antepone `Documento > Sección` al texto embebido ("Ejemplo 2" solo no
dice de qué tema es). Con E5 los prefijos son obligatorios: sin ellos baja la
calidad; consulta y documentos deben usar el mismo modelo.

*Límites del benchmark:* 20 preguntas escritas por quien implementa; sirve para
descartar lo claramente peor, no para afinar decimales (eso es la Fase 2).

---

## 5. Base vectorial (`rag/store.py`)

**ChromaDB `PersistentClient`** en `avance2/chroma_db/`, distancia **coseno**
(`hnsw:space=cosine`; distancia = 1 − similitud; 0 idéntico, 2 opuesto).
Alternativas: FAISS (más bajo nivel, sin metadatos), Qdrant/Weaviate (servidor),
`pgvector`. Chroma es embebido, persistente y filtra por metadatos, y el índice
es solo una carpeta tuya.

Metadatos por chunk: `fuente`, `documento`, `seccion`, `pagina` (0 = el formato
no tiene páginas, porque Chroma no admite `None`), `chunk_id`, `tipo`, `formato`,
`n_chunk`, `excede_tokens`. `tipo` (teoría, definición, formulario, ejemplo,
procedimiento, ejercicio, respuesta) se deduce del encabezado más específico:
distribución actual `ejemplo 46, formulario 10, ejercicio 9, teoría 9,
procedimiento 8, respuesta 4, definición 2`. La colección guarda una **huella**
(hash de contenido + parámetros) para saber si un índice existente sigue vigente.
`embedding_function=None` evita que Chroma use su propio modelo y descargue otro.

---

## 6. Recuperación (`rag/retrieve.py`)

`top_k = 4` (configurable). Se vectoriza la consulta con `query: ` y se buscan los
vecinos por coseno. `UMBRAL_DISTANCIA = None` **a propósito**: no se fija a ojo.

Prueba por CLI (5 consultas, `python -m rag.retrieve …`):

| Consulta | ¿En corpus? | top-1 | distancia |
|---|---|---|---|
| raíz de a²−x² → sustitución | sí | tabla "Casos estándar" ✔ | 0,1133 |
| partes y regla ILATE | sí | "Algoritmo General" (ILATE) ✔ | 0,1061 |
| simplificar una fracción antes de integrar | sí | "Algoritmo … Simplificación" ✔ | 0,1155 |
| volumen por discos | **no** (tema afín) | "Ejercicios de identificación de u y du" | 0,1670 |
| capital de Francia / Torre Eiffel | **no** | ejemplo de la Torre CN (cinemática) | 0,2383 |

Lo que enseñan estos datos (importante para calibrar el umbral en la Fase 2):

- Las distancias de E5 se concentran en un rango estrecho (0,09–0,24): hay separación, pero **pequeña**. En el benchmark de 20 preguntas el peor top-1 *dentro* del corpus es **0,150** y el mejor *fuera* es **0,151** (margen 0,001): cualquier umbral hoy sería una apuesta. Hacen falta más preguntas (Fase 2.3).
- **La distancia mide cercanía temática, no corrección.** Una consulta con fórmula dicha en palabras ("dx entre x² por raíz de 9−x²") obtuvo distancia 0,099 con un fragmento *equivocado* y no encontró el Ejemplo 1. Un umbral solo descartará lo claramente ajeno; no sustituye a la regla de abstención del prompt.
- Los fallos "rango > 1" del benchmark son sobre todo **vecinos legítimos** (la "Estrategia sen^m cos^n" para "potencias pares de coseno"), no errores de recuperación.

---

## 7. Generación (`rag/prompts.py`, `rag/generate.py`)

**Capa de LLM (`rag/llm.py`).** Proveedor por defecto: **Anthropic** (SDK oficial
`anthropic`), con `LLM_PROVIDER` / `LLM_MODEL` (tutor: `claude-haiku-5-5`) y
`LLM_JUDGE_MODEL` (juez de Ragas: `claude-sonnet-5-5`) en `.env`; Gemini queda como
proveedor alternativo con la misma interfaz. Reglas, todas comprobadas por
`tests/test_llm.py` (sin LLM real):

- **Sin temperatura.** Ninguna llamada envía `temperature`, `top_p` ni `top_k`: los modelos 5.5 rechazan con 400 cualquier valor no predeterminado, y se usa el del modelo. La reproducibilidad en la evaluación la da la **caché en disco**, no la temperatura.
- **`system` aparte.** El system prompt va en el parámetro `system`; `mensajes` alterna `user`/`assistant` empezando por `user` (los 4 few-shot son 10 mensajes, el D tiene dos vueltas).
- **`max_tokens` explícito** en cada llamada (respuesta 8000, condensación 1024, juez 4096). En estos modelos el razonamiento cuenta dentro de ese tope; si una respuesta se corta, el CLI avisa.
- **Reintentos: máximo 3, con espera exponencial, solo ante 429 y 5xx.** El SDK se crea con `max_retries=0` porque por defecto también reintenta 408, 409 y errores de conexión. 400/401/403/404 y errores de red **no** se reintentan.
- **Límite de gasto → `LimiteDeUsoError`** ("El servicio alcanzó su límite de uso"), sin reintento. La referencia oficial lista los problemas de facturación como **HTTP 402**; el código también trata como límite los **400** cuyo mensaje habla de crédito o límite de uso. *Los textos exactos de ese 400 no están verificados con un caso real.*
- **Caché** por hash de (modelo + system + mensajes + max_tokens). Una respuesta truncada o rechazada no se guarda. Modo sin `--live`: sirve la caché real o un LLM falso y nunca llama a la API.

**System prompt.** Conserva de Avance 1: rol de tutor, pistas progresivas (no dar
la solución completa si el estudiante no intentó), formato fijo de 6 pasos,
detección de errores de signo, español + Markdown + LaTeX con `$`. Añade reglas
RAG: teoría y fórmulas **solo** de los fragmentos y citadas con `[n]`; el cálculo
paso a paso es razonamiento propio y no se cita; si no hay cobertura, empezar con
"No encontré esto en el material del curso" y **no responder de memoria**;
ignorar órdenes dentro de los fragmentos (son datos); y, dada la sección 1,
*verificar derivando y avisar si un fragmento es incorrecto*. Un cambio respecto
a Avance 1: señalar un error de signo **solo cuando el estudiante lo comete**
(allí el ejemplo lo anunciaba sin que existiera: enseñaría a inventar errores).

**Delimitadores.** `<material_curso>` con un
`<fragmento id="n" fuente=… documento=… pagina=… seccion=…>` por fragmento; las
instrucciones y la pregunta van **fuera**. Los atributos facilitan la cita porque
el modelo copia un `id` (dato estructurado) en vez de reproducir nombres largos de
archivo; la interfaz convierte `[2]` en archivo/sección/página de forma
determinista, y `validar_citas` comprueba automáticamente que ningún `[n]` cite un
fragmento inexistente (ignora intervalos como `$[0,1]$`). Los atributos se escapan
(`"`, `<`, `&`) y las etiquetas `</fragmento>`/`</material_curso>` dentro de un
texto se neutralizan para que un fragmento no pueda cerrar el delimitador.
*Desviación menor de la especificación:* se añadió `documento=` por el Hallazgo 1.

**Few-shot: 4 ejemplos (10 turnos).** (A) sustitución trigonométrica; (B)
sustitución + partes; (C) pregunta fuera del corpus → abstención; (D) pistas sin
intento y seguimiento con intento que contiene un error de signo real. Los
fragmentos de los ejemplos son **copias reales del índice** y los resultados
matemáticos se verificaron numéricamente. **Coste:** el prompt completo midió
**11 369 tokens de entrada con Claude** (7 306 con Gemini: otro tokenizador) en el
turno 1 de la demo; los ejemplos son ~15 000 caracteres. Es el precio de ejemplos
de calidad; reducirlo (p. ej. quitar el B) es una palanca si la cuota aprieta.

**Condensación del seguimiento.** Antes de recuperar, si hay historial, se
reescribe la pregunta como autónoma (`<historial>` + `<pregunta_seguimiento>`,
últimos 4 turnos recortados a 600 caracteres). La recuperación usa
la condensada; la generación recibe la pregunta **original** y los **turnos
reales** del historial (últimos 6). Si la condensación falla, se busca con la
pregunta original y se avisa.

### Demo conversacional con Haiku 5.5 (salida real, 3 turnos)

| Turno | Qué pasó | Tokens ent./sal. | Tiempo |
|---|---|---|---|
| 1. Intento + "solución completa" (`∫dx/√(4x²+1)`) | Recuperó el Ejemplo 2 de sust. trigonométrica (dist. 0,087); 6 pasos completos, resultado y verificación derivando correctos | 11 369 / 2 072 | 59,1 s |
| 2. Seguimiento: "¿Y si en lugar de 4x² fuera 9x²?" | Condensó la pregunta y recuperó los ejemplos de sust. trigonométrica; **dio pistas y no la solución** (el estudiante no intentó), como exige la regla | 12 732 / 1 219 | 9,8 s |
| 3. Volumen por discos (fuera de corpus) | Empezó exactamente con "No encontré esto en el material del curso" y no respondió de memoria | 13 625 / 493 | 3,5 s |

Coste estimado con tarifas de lista de Haiku 5.5 (0,10 / 0,50 USD por millón de
tokens): ≈ 0,006 USD los tres turnos, sin contar las 2 llamadas de condensación
(no medidas). Es una estimación, no un dato de facturación.

**Defectos observados (reales, a vigilar en la Fase 2):**

1. **Cita incorrecta que el validador no detecta.** En el turno 1, "por Pitágoras … [3]" cita el fragmento [3], que es la tabla de *respuestas* de Simplificación algebraica, no un texto sobre Pitágoras. `citas_invalidas` solo comprueba que el número exista; que la cita *respalde* la frase es lo que mide `faithfulness`.
2. **Condensación imperfecta.** En el turno 2 la pregunta autónoma conservó la sustitución anterior (`x = tanθ/2`) pese a que con `9x²` correspondería `tanθ/3`. La recuperación no se vio afectada, pero es una distorsión.
3. **Afirmación sin base.** En el turno 1 el tutor supuso "si en tu intento no calculaste `dx`…", algo que el estudiante no dijo.
4. **Formato de cita** no estándar en el turno 3 (`[1 de sustitución]`).
5. **Latencia.** 59 s en el turno 1 (razonamiento adaptativo por defecto del modelo); relevante para el chat de la Fase 3.

Estos defectos no se han corregido: son evidencia para la evaluación y para la
decisión de qué iterar (prompt vs. generación).

---

## 8. Evaluación con Ragas (Fase 2)

### Corridas y resultados (reales)

| Corrida | Tutor | Juez | `top_k` | Nota |
|---|---|---|---|---|
| `piloto` | Haiku 5.5 | Haiku 5.5 | 4 | 3 preguntas: comprueba el cableado |
| `base_juez4096` | Haiku 5.5 | Haiku 5.5 | 4 | **defectuosa** (5 NaN por el tope de tokens del juez) |
| `base` | Haiku 5.5 | Haiku 5.5 | 4 | desarrollo (juez barato) |
| `iter1_topk8` | Haiku 5.5 | Haiku 5.5 | 8 | iteración, juez barato |
| **`final_k4`** | Haiku 5.5 | **Sonnet 5.5** | 4 | corrida final: línea base |
| **`final_k8`** | Haiku 5.5 | **Sonnet 5.5** | 8 | corrida final: tras la iteración |

**Línea base (`final_k4`, 18 preguntas del corpus, IC 95 % por bootstrap):**

| Métrica | Media | IC 95 % | Desv. est. |
|---|---|---|---|
| Faithfulness | **0,546** | [0,404, 0,689] | 0,313 |
| Answer relevancy | 0,890 | [0,777, 0,952] | 0,224 |
| Context precision | 0,718 | [0,519, 0,894] | 0,406 |
| Context recall | 0,685 | [0,481, 0,880] | 0,431 |

Por tipo (faithfulness / recall): definición 0,79 / 0,92 · procedimiento 0,46 / 0,60 ·
ejercicio 0,39 / 0,62 · seguimiento 0,41 / 0,44. Con el juez barato (Haiku) la base fue
0,540 / 0,825 / 0,718 / 0,667: **los dos jueces coinciden en la media de faithfulness**.
Fragmento LaTeX listo para pegar: `eval/results/final_k4.tabla.tex`; gráficos:
`final_k4.png`, `compare_final_k4_final_k8.png`; informe completo: `final_k4.analisis.md`.

**Fuera de corpus (aparte de Ragas).** 5 preguntas: **abstención correcta 5/5 (1,0)**,
**alucinación 0,0**. Abstención *indebida* (respondió "no encontré" a una pregunta del corpus):
1 de 18 (5,6 %) con `top_k = 4`, 0 de 18 con `top_k = 8`. Citas a fragmentos inexistentes: 0.

### Calibración del umbral de distancia (2.3)

Distancia coseno del mejor fragmento (top-1), con la pregunta efectivamente buscada:

- **Dentro** del corpus (n = 18): mín 0,0887 · mediana 0,1048 · máx 0,1422.
- **Fuera** del corpus (n = 5): mín 0,1313 · mediana 0,1511 · máx 0,1820.
- AUC = **0,97**, pero **margen = −0,0109: las distribuciones se solapan** (ningún umbral separa sin errores).
- Mejor umbral por exactitud balanceada: **0,1295** (0,94): bloquea las 5 de fuera, pero también **2 preguntas válidas** ("¿En qué consiste el método de sustitución…?", 0,1422, y "¿De dónde sale la fórmula de integración por partes?", 0,1327).
- Umbral conservador **≈ 0,145** (justo por encima del máximo dentro): no bloquea ninguna válida y evita llamar al modelo en 3 de 5 preguntas ajenas (0,1511; 0,1669; 0,1820); las otras 2 (0,1313; 0,1405) pasan al LLM, que se abstuvo correctamente.

**Recomendación para la Fase 3 (decisión abierta):** como la abstención del propio LLM ya es 5/5, el
umbral **no debe usarse para decidir la respuesta**; solo como filtro conservador de coste (≈ 0,145)
o no usarse. Un umbral estricto (0,1295) rechazaría preguntas legítimas de estudiantes, que es peor
que una llamada de más. **Cautela:** solo hay 5 preguntas fuera de corpus; el máximo "dentro"
(0,1422) cambiaría con más preguntas.

### Análisis (2.4): ¿qué métrica es la más baja y a qué se debe?

La más baja es **faithfulness (0,546)**. Atribución con evidencia (`final_k4`):

1. **Recuperación (causa principal).** En **4 de 18** preguntas (#7, 11, 12, 15; todas consultas largas de "estudiante que ya intentó") la sección correcta no llega al top-4. En esas 4: faithfulness **0,14** y recall **0,12**; en las 14 donde sí llega: faithfulness **0,66** y recall **0,85**. El benchmark gratuito `eval/bench_retrieval.py` sitúa la sección correcta de esas 4 en los puestos **8, 12, 5 y 5**. Las consultas largas con mucha notación se parecen más a listas de integrales ("Análisis de selección", "Respuestas a los ejercicios propuestos") que a la sección que explica el método.
2. **Generación (techo residual).** Con la sección correcta presente, las respuestas de solución completa siguen en faithfulness ≈ **0,63–0,66**: el cálculo paso a paso (álgebra, derivadas, verificación) es razonamiento propio del tutor y **no está en los fragmentos** (justo lo que la regla de uso del material pide). Es un efecto de medir faithfulness en un tutor que calcula, no un defecto a eliminar.
3. **Chunking: no es la causa.** Con 1000/150 la sección entra en el top-4 en 15/18 (+1) y con 600/100 en 15/18; el modo de texto `verbal` no mejora.
4. **Interacción con la pedagogía.** El tutor respondió con *pistas* a 2 preguntas del dataset (#3 definición, #12 ejercicio) pese a que #12 dice que ya intentó y pide el resultado: el estudiante no mostraba su trabajo. Esas respuestas puntúan bajo en `answer_relevancy` (0,00 en #12) aunque la conducta sea defendible. **El dataset no se modificó** (aprobado).

**Ruido (qué es demasiado pequeño para concluir).** Con 18 preguntas y un juez LLM los IC son anchos
(≈ ±0,14 en faithfulness; ±0,2 en recall/precision). Ninguna diferencia entre corridas de esta fase
es estadísticamente distinguible: todos los IC se solapan.

### Dataset (paso 2.1) — `eval/dataset.jsonl`, **aprobado por el usuario tal cual**

23 preguntas redactadas leyendo el corpus real: 6 `definicion`, 5 `procedimiento`,
4 `ejercicio`, 3 `seguimiento` (con `historial`) y 5 `fuera_de_corpus`. Campos:
`question`, `ground_truth`, `tipo`, `fuente_esperada` (nombre de archivo; vacío en
fuera de corpus), más `seccion_esperada` e `historial` (opcionales).

- Por la regla de pedagogía, las 9 de `procedimiento`/`ejercicio` están escritas como estudiante que **ya intentó** y pide la solución completa (si no, el tutor responde con pistas y no habría solución que evaluar).
- Los `seguimiento` son preguntas conceptuales con un intercambio previo, para ejercitar la condensación sin chocar con la regla de pistas.
- Los 5 `fuera_de_corpus` (discos, área entre curvas, Riemann/sigma, ecuación diferencial separable, teorema fundamental/integral definida) se comprobaron **ausentes** del texto del corpus. Su `ground_truth` es la abstención correcta.
- Todos los resultados matemáticos de los `ground_truth` provienen de ejemplos/respuestas del corpus que pasaron la auditoría numérica (sección 1). No se usó ninguna de las 2 respuestas pendientes de partes.

**Comprobación preliminar de recuperación** (local, sin LLM; *no es una métrica*):
el archivo esperado está en el top-4 en 17 de 18 preguntas del corpus. La excepción
es la pregunta de `∫dx/(1+sen x)` (simplificación, ej. f), cuyo archivo no aparece. Tres
preguntas más solo lo traen en el puesto 3-4. Las distancias del top-1 de las
preguntas fuera de corpus (0,131–0,182) se solapan con las de algunas del corpus
(0,089–0,154): confirma que **el umbral no puede fijarse todavía**.

`eval/run_eval.py` y `eval/juez.py` se probaron **sin LLM real** (`tests/test_eval.py`)
y después con corridas reales (sección "Corridas y resultados").

**Cómo se conecta el juez Anthropic a Ragas 0.4.3** (verificado en el código
instalado):

- `llm_factory(provider="anthropic")` **no sirve**: usa `instructor`, que por defecto fuerza `tool_choice` → `{"type":"tool"}` (Sonnet 5.5 lo rechaza con 400), y Ragas envía temperatura 0,01 y `top_p` 0,1 (los modelos 5.5 los rechazan).
- Se usa el **wrapper de LangChain** (`langchain-anthropic` + `LangchainLLMWrapper`): las métricas clásicas piden JSON como texto, sin herramientas forzadas. Es una API **deprecada** en 0.4.3 (por eso `ragas` está fijado en esa versión). Hay que heredar de `ragas.llms.base.LangchainLLMWrapper`: el nombre exportado por `ragas.llms` es un envoltorio de deprecación del que no se puede heredar.
- Ragas **asigna** `temperature` al objeto antes de cada llamada síncrona, y `langchain-anthropic` lanza `ValueError` si un modelo 5.5 recibe muestreo no predeterminado. El juez (`ChatAnthropicJuez`) ignora esas asignaciones y limpia el payload: el muestreo no puede fijarse.
- `RunConfig` de Ragas reintenta por defecto **cualquier** excepción 10 veces (incluido un límite de gasto). Se ajusta a 3 reintentos, solo `RateLimitError`/`InternalServerError`, 2 workers. `max_retries` de Ragas cuenta intentos totales, de ahí el `+1`. Un límite de uso se anota y la corrida termina con el mensaje claro (Ragas, si no, lo convertiría en puntajes `NaN` silenciosos).
- `answer_relevancy` usa el mismo modelo E5 local, con prefijo `query:` en ambos lados.

**Control de gasto de `run_eval.py`:** sin `--live` no hay llamadas (caché real o LLM
falso; se guarda como `*.OFFLINE.json`, nunca como resultado válido); con `--live`
todo pasa por la caché en disco, así que relanzar tras un corte no repite lo pagado;
el juez es `LLM_MODEL` salvo con `--juez-final` (`LLM_JUDGE_MODEL`); `--limite N`
usa solo las primeras N preguntas; progreso por pregunta; ante cualquier error la
corrida se detiene y lo explica. Las preguntas `fuera_de_corpus` se miden aparte
(tasa de abstención correcta y de alucinación). **Comprobado en vivo:** Ragas puntúa con el juez Anthropic (piloto de 3 preguntas, 12 de 12
tareas). La primera corrida completa reveló un fallo real: con `MAX_TOKENS_JUEZ = 4096` el juez
se quedaba sin tokens (`LLMDidNotFinishException`: el razonamiento del modelo + la descomposición
de una respuesta de ~2 000 tokens) y Ragas devolvía **NaN silencioso en faithfulness para las 5
respuestas de solución completa**. Se subió a 16 000 y se archivó la corrida defectuosa como
`base_juez4096`. La caché del juez no incluye `max_tokens` en su clave (solo guarda respuestas
completas), por lo que reutiliza llamadas idénticas entre corridas.

## 9. Iteración (Fase 2.5) — un solo parámetro: `top_k` 4 → 8

**Por qué este parámetro.** La métrica más débil (faithfulness) cae a 0,14 exactamente cuando la
sección correcta no entra en el contexto, y recall es la segunda más baja; la guía orientativa
("recall bajo → chunking / `top_k`") coincide. Antes de gastar en el juez se midió, gratis, qué
rescata más preguntas (`eval/bench_retrieval.py`, sección correcta en el top-k, de 18):

| Variante | top-4 | top-6 | top-8 |
|---|---|---|---|
| actual (limpio, 1800/200) | 14 | 16 | 17 |
| texto `verbal` | 13 | 16 | 17 |
| texto crudo | 14 | 16 | 16 |
| chunk 1000/150 | 15 | 16 | 17 |
| chunk 600/100 | 15 | 16 | 17 |

Cambiar el chunk o el modo de texto apenas mueve (±1); subir `top_k` rescata 3 de las 4. Se cambió
**solo** `top_k` (mismo dataset, mismo juez, mismo tutor).

**Resultado (`final_k4` → `final_k8`, juez Sonnet 5.5):**

| Métrica | Antes | Después | Δ | ¿Dentro del ruido? |
|---|---|---|---|---|
| Faithfulness | 0,546 | 0,605 | +0,059 | sí |
| Answer relevancy | 0,890 | 0,943 | +0,053 | sí |
| Context precision | 0,718 | 0,697 | **−0,020** | sí |
| Context recall | 0,685 | 0,731 | +0,046 | sí |

Con el juez barato el resultado fue de la misma forma (faithfulness 0,540 → 0,586; recall 0,667 →
0,722; precision 0,718 → 0,680). **Lo que empeoró:** context precision baja en las dos corridas
(más fragmentos = más ruido), y el coste sube un **+17 % de tokens de entrada por pregunta**
(11 294 → 13 194 de media, tutor Haiku). **Lo que no mejoró:** la #11 (`∫dx/(1+sen x)`) tiene su
sección en el puesto 12: con `top_k = 8` el tutor pasó de abstenerse a dar una solución, pero con
faithfulness 0,00 (la resolvió sin apoyo en los fragmentos); y la #15 sigue en faithfulness 0,06.
En las 4 preguntas objetivo (corrida con juez barato) faithfulness 0,17 → 0,30 y recall 0,00 →
0,33; en las otras 14 casi no cambió (0,65 → 0,67 y 0,86 → 0,83).

**Conclusión honesta:** la dirección es coherente entre los dos jueces y con el benchmark de
recuperación, pero **con 18 preguntas no se puede afirmar una mejora**: todas las diferencias
caen dentro del ruido. La evidencia sí apoya que el techo de faithfulness del tutor, una vez
recuperada la sección correcta, es ≈ 0,64 por el cálculo propio. Siguiente candidato, si se
itera: recuperar con una versión *recortada* de las consultas largas (quitar la narración del
intento) en vez de subir más `top_k`.

Comparaciones guardadas: `eval/results/compare_final_k4_final_k8.{md,png}` (final) y
`compare_base_iter1_topk8.{md,png}` (desarrollo).

**Coste real de la Fase 2** (tokens medidos por los contadores, a tarifas de lista; estimación, no
facturación): juez Haiku ≈ 0,13 USD por corrida completa; juez Sonnet ≈ 1,6 USD (`final_k4`) y
1,8 USD (`final_k8`); tutor ≈ 0,002 USD por pregunta. Total aproximado ≈ 4 USD.

---

## 10. App web y despliegue (Fase 3)

### La app (`app.py`)

Una sola página de Streamlit que reutiliza el pipeline sin tocarlo (`rag.generate.responder`).
Decisiones:

- **Chat con historial** en `st.session_state`: cada turno se guarda como texto
  (pregunta y respuesta); el historial real se reenvía al modelo y la condensación de
  seguimientos funciona como en la CLI.
- **LaTeX**: `st.markdown` renderiza `$…$` y `$$…$$` con KaTeX; el prompt ya pide ese formato.
  *Defecto encontrado en la prueba real:* Markdown junta los saltos de línea simples, así que
  el título de cada paso quedaba pegado a su texto. `md()` convierte el salto simple en salto
  duro (los párrafos no cambian). Hay un test (`test_saltos_de_linea_simples_se_conservan`).
- **"Fuentes"**: expander bajo cada respuesta con `[n]`, documento › sección, distancia
  coseno y los primeros 700 caracteres del fragmento (el modelo recibe el fragmento entero).
  Se muestra incluso cuando el tutor se abstiene, para que se vea qué se recuperó.
- **Protección de cuota** (la app es pública): máximo **15 preguntas por conversación**
  (`APP_MAX_PREGUNTAS`), **1 500 caracteres** por pregunta, `max_tokens` explícito en cada
  llamada, sin caché en disco (no guarda preguntas). Botón **Nueva conversación**. Contador
  de preguntas y de tokens en la barra lateral (los tokens son los de la generación; no
  incluyen la llamada de condensación). *Esto no impide que alguien abra muchas
  conversaciones:* la protección real es el límite de gasto de la consola del proveedor.
- **Errores**: límite de gasto → mensaje fijo "El servicio alcanzó su límite de uso";
  cualquier otro error → mensaje genérico (el detalle solo va al log del servidor, nunca
  al usuario, para no filtrar rutas ni texto de la API).
- **Secretos**: `ANTHROPIC_API_KEY` se lee del entorno / `.env` y, si no está, de `st.secrets`
  (`rag/llm.py::_secreto`). Si falta, la app avisa y no muestra el chat.
- **Índice**: no se sube al repo. `ingest.asegurar_indice()` lo reconstruye en el primer
  arranque si no existe o su huella no coincide con el corpus y los parámetros.
- **Telemetría**: `.streamlit/config.toml` con `gatherUsageStats = false`. También
  `fileWatcherType = "none"`: el vigilante de archivos de Streamlit recorría los módulos de
  `transformers` e inundaba el log con errores de `torchvision` (no instalado).
- **`top_k` de la app = 8** (`APP_TOP_K` en `config.py`). *Decisión abierta:* la Fase 2 no
  pudo demostrar que 8 supere a 4 (dentro del ruido); elegí 8 porque recupera la sección
  correcta en 17/18 preguntas (4: 14/18) y la abstención indebida baja de 1 a 0, a cambio de
  ≈ 17 % más tokens de entrada. Se revierte con una línea.
- **Umbral de distancia: no se usa** (`UMBRAL_DISTANCIA = None`). Los datos de la Fase 2
  muestran solapamiento y el LLM ya se abstiene bien (5/5). Un filtro conservador en ≈ 0,145
  solo ahorraría llamadas; si lo quieres, es una línea en `config.py`.

**Tests sin LLM real:** `python -m tests.test_app` (9 tests con `streamlit.testing.v1.AppTest`
y LLM falso: carga, un turno con fuentes, seguimiento, límite de preguntas, reinicio, límite
de uso, error interno sin filtrar detalles, falta de clave, saltos de línea).

**Prueba local de 4 turnos con Haiku 5.5** (real): ver tabla en `docs/DESPLIEGUE.md` §5. Resultado:
los 4 turnos correctos (solución verificada, seguimiento con contexto, abstención fuera de
corpus con la frase exacta, definición con citas). Memoria del proceso ≈ **1,34 GB**; arranque en
frío ≈ **70 s**.

### Dónde desplegar: hallazgo que cambia el plan

Verificado en la documentación vigente (2026-10-16): **Hugging Face Spaces ya no ofrece el
SDK `streamlit`** (solo `gradio`, `docker`, `static`) y los Spaces Docker/Gradio **requieren un
plan de pago** para crearse. **Streamlit Community Cloud** es gratis, soporta Streamlit de
forma nativa, da ≈ 2,7 GB de RAM (cifra de febrero de 2024) y duerme la app a las 12 h sin
tráfico. **Recomendación: Streamlit Community Cloud.** Guía paso a paso, qué se publica y
riesgos no verificados: [`DESPLIEGUE.md`](DESPLIEGUE.md). **Decisión tuya:** ¿Community Cloud
(gratis) o Hugging Face (pago, requiere un Dockerfile que aún no escribí)?

**Lo que se publicaría** si el repositorio es público: los 6 archivos de `corpus/`, los
ejemplos del prompt (copias del corpus), el dataset y los resultados de evaluación. Si los
apuntes tienen derechos de terceros, no uses un repositorio público.

---

## 11. Privacidad

**Qué sale de tu máquina hacia el proveedor del LLM** (Anthropic por defecto; única
salida de datos del pipeline de consulta), en cada pregunta:

1. La pregunta del estudiante.
2. El historial (últimos 6 turnos reales; y, en la condensación, los últimos 4 recortados a 600 caracteres).
3. Los `top_k` fragmentos recuperados (4 por defecto en la CLI y en la evaluación base; **8 en la app**).
4. El system prompt y los **4 ejemplos few-shot**, que son texto fijo del proyecto e **incluyen copias de fragmentos reales del corpus** (se envían en *todas* las llamadas).

**No sale:** el corpus completo, los archivos, el índice de Chroma ni los embeddings
(se calculan y almacenan localmente). **En la evaluación (Fase 2)** además se
enviarán al LLM juez la pregunta, los contextos recuperados, la respuesta y la
referencia.

**Otras comunicaciones de red:** descarga de los pesos del modelo de embeddings
desde Hugging Face (solo pesos; el modelo se carga después desde caché local, sin
red). Telemetría desactivada: Chroma (`anonymized_telemetry=False` + variable
`ANONYMIZED_TELEMETRY`) y Ragas (`RAGAS_DO_NOT_TRACK=true`, verificado en su
código fuente, que de otro modo envía a `t.explodinggradients.com`). Streamlit
(`gatherUsageStats = false` en `.streamlit/config.toml`).

**Claves:** solo por variable de entorno / `.env` (ignorado por git) o secrets de la
plataforma; nunca en el repo.

**Caché local del LLM** (`.cache_llm/`, ignorada por git): guarda en disco las
respuestas del modelo, que incluyen texto del corpus y las preguntas de la
evaluación. Es local, pero si el disco se comparte conviene borrarla. La app
pública **no** la usa (no guarda preguntas de estudiantes).

**⚠ A COMPROBAR POR TI (no verificado):** las condiciones de uso de datos de la API
de Anthropic (y, si usas el proveedor alternativo, las de la API de Gemini)
dependen del tipo de cuenta y plan: retención de los datos enviados, uso para
mejorar modelos, revisión humana. **No se han comprobado en los términos vigentes
y no se afirma nada.** Revísalos en la documentación oficial del plan que
usas antes de enviar material del curso o datos de estudiantes. Además, si el
material tiene derechos de autor de terceros, revisa si puedes publicarlo en un
repositorio.
