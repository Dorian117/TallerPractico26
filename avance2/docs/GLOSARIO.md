# Diccionario de términos

Orden alfabético. Cada entrada dice **qué es** y, cuando ayuda, **dónde aparece en este
proyecto**. Los términos entre «→» remiten a otra entrada. Para el recorrido del código,
ver [`GUIA_CODIGO.md`](GUIA_CODIGO.md).

---

## A

**Abstención.** Cuando el tutor responde «No encontré esto en el material del curso» en lugar
de contestar de memoria. Es la conducta correcta ante una pregunta fuera del temario.
*Proyecto:* `eval/run_eval.py::se_abstiene`; resultado: 5/5 preguntas fuera de corpus.

**Abstención indebida.** Abstenerse ante una pregunta que **sí** estaba en el corpus. Un
error por exceso de prudencia. *Proyecto:* 1 de 18 con `top_k = 4`, 0 de 18 con `top_k = 8`.

**Alucinación.** Cuando un LLM afirma algo falso o inventado con aparente seguridad. Aquí se
mide como «pregunta fuera del corpus que el tutor **no** rechazó» (resultado: 0).

**Answer relevancy.** Métrica de → Ragas: ¿la respuesta es pertinente a la pregunta? De 0 a 1.

**API.** Interfaz para que un programa use un servicio ajeno; aquí, la de Anthropic para
enviar un mensaje a Claude y recibir la respuesta.

**API key (clave de API).** Contraseña que identifica tu cuenta ante la API y la que **cobra**.
Va solo en `.env` o en los *secrets* de la plataforma; **nunca** en el repositorio.

**Átomo (indivisible).** En el → chunking, una fórmula (`$…$`, `$$…$$`, `\[…\]`,
`\begin…\end`) que no se puede cortar. *Proyecto:* `chunking.py::_enmascarar`.

**AUC.** Número entre 0 y 1: probabilidad de que una pregunta **del corpus** tenga el mejor
fragmento más cerca que una pregunta **de fuera**. 1 = separación perfecta, 0,5 = azar.
*Proyecto:* AUC = 0,97 (pero con solapamiento, así que el → umbral no sirve).

## B

**Bloque.** Trozo de texto con su sección y página, tal como lo entrega un lector.
*Proyecto:* `readers.Bloque`.

**Bootstrap.** Técnica para estimar cuánto «ruido» tiene una media: se remuestrea muchas veces
los mismos datos con reemplazo y se mira cuánto varía el resultado. Genera el → intervalo de
confianza. *Proyecto:* `analisis.ic_bootstrap` (5 000 remuestreos).

## C

**Caché.** Guardar un resultado para no repetir el trabajo (y el gasto). *Proyecto:* la
caché de respuestas del LLM en `.cache_llm/` (clave = hash del prompt); la **app no la usa**.
No confundir con `st.cache_resource` (→ Streamlit).

**Chroma / ChromaDB.** Base de datos de → vectores. Aquí es una carpeta local (`chroma_db/`)
sin servidor. *Proyecto:* `rag/store.py`.

**Chunk (fragmento).** Trozo de los apuntes de tamaño acotado que se indexa y se recupera.
Es la unidad mínima que el buscador devuelve. *Proyecto:* 88 chunks; `chunking.Chunk`.

**Chunking.** El proceso de trocear los documentos en → chunks. *Proyecto:* `rag/chunking.py`.

**Citación `[n]`.** Marca que el tutor pone en su respuesta para indicar de qué → fragmento
sacó una fórmula o definición. `n` es el `id` del fragmento en ese turno.

**Citas inválidas.** Citas `[n]` a un fragmento que no existe en ese turno. *Proyecto:*
`prompts.citas_invalidas`. **Límite:** solo comprueba que el número exista, no que el
fragmento respalde lo dicho.

**CLI.** Interfaz de línea de comandos (por consola). *Proyecto:* `python -m rag.cli`.

**Colección.** En Chroma, el «contenedor» de vectores, equivalente a una tabla. *Proyecto:*
se llama `calculo2`.

**Condensación.** Reescribir una pregunta de seguimiento («¿y si fuera 9x²?») como una
pregunta **autónoma** que se entienda sin la conversación, para poder buscarla bien.
*Proyecto:* `generate.condensar_pregunta`.

**Contexto (en Ragas).** Los fragmentos recuperados para una pregunta (`retrieved_contexts`).

**Context precision.** Métrica de → Ragas: ¿los fragmentos relevantes aparecen **arriba** del
ranking? Baja si se meten fragmentos inútiles delante.

**Context recall.** Métrica de → Ragas: ¿lo recuperado contiene lo necesario para la respuesta
de referencia?

**Corpus.** El conjunto de documentos que alimenta el sistema. *Proyecto:* 6 archivos `.txt`
con LaTeX en `corpus/`.

## D

**Dataclass.** Clase de Python pensada para guardar datos (`@dataclass`). *Proyecto:*
`Bloque`, `Chunk`, `Fragmento`, `Respuesta`, `RespuestaLLM`, `Turno`.

**Dataset (de evaluación).** Lista de preguntas con su respuesta de referencia para medir la
calidad. *Proyecto:* `eval/dataset.jsonl` (23 preguntas).

**Delimitadores.** Etiquetas que separan el material de las instrucciones dentro de un prompt
(`<material_curso>`, `<fragmento>`). Ayudan al modelo a no confundir datos con órdenes.

**Deprecado (deprecated).** Función que aún funciona pero se eliminará en el futuro.
*Proyecto:* el envoltorio de Ragas para LangChain; por eso `ragas` se fija en 0.4.3.

**Distancia coseno.** Medida de lo lejos que están dos → vectores: `1 − similitud coseno`.
**0 = idéntico**, 1 = sin relación, 2 = opuesto. En este proyecto, valores típicos: dentro
del corpus ≈ 0,09–0,14; fuera ≈ 0,13–0,18.

**Docker.** Tecnología que empaqueta una aplicación con su entorno en un «contenedor».
Hugging Face Spaces lo usa para apps que no son Gradio.

## E

**E5 (`multilingual-e5-small`).** El modelo de → embeddings que usa el proyecto: pequeño,
multilingüe y local. Exige los prefijos `query: ` y `passage: `.

**Embedding.** Vector de números que representa el significado de un texto; textos de
significado parecido quedan cerca. *Proyecto:* `rag/embeddings.py`; dimensión 384.

**Entorno virtual (venv).** Carpeta con una copia aislada de Python y sus librerías, para que
las versiones del proyecto no choquen con las de otros. *Proyecto:* `avance2/.venv`.

## F

**Faithfulness (fidelidad).** Métrica de → Ragas: ¿lo que afirma la respuesta está respaldado
por los fragmentos? *Proyecto:* 0,546 con `top_k = 4` (la más baja).

**Few-shot.** Dar al modelo **ejemplos** resueltos dentro del prompt para que imite su formato
y conducta. *Proyecto:* 4 ejemplos (10 mensajes) en `prompts.EJEMPLOS_FEW_SHOT`.

**Fragmento.** → Chunk. En el código, `retrieve.Fragmento` es un chunk ya recuperado, con su
distancia y su posición.

**Fuera de corpus.** Pregunta cuya respuesta no está en los apuntes (p. ej. «volumen por
discos»). El tutor debe abstenerse.

## G

**Git / commit / push.** Git guarda el historial del código; un *commit* es una «foto» del
proyecto; *push* la sube a GitHub.

**Ground truth (referencia).** Respuesta correcta escrita de antemano para una pregunta del
dataset; sirve de patrón de comparación.

## H

**Haiku / Sonnet.** Modelos de Claude: Haiku 5.5 (rápido y barato) es el **tutor**; Sonnet
5.5 (más capaz y caro) es el **juez** de las corridas finales.

**HNSW.** Estructura de índice que permite buscar vecinos cercanos en un conjunto grande de
vectores sin compararlos todos. Chroma la usa internamente.

**Huella (del índice).** Hash que resume el contenido del corpus y los parámetros con que se
construyó el índice; si cambia, el índice está «viejo» y se reconstruye. *Proyecto:*
`store.huella`, `ingest.asegurar_indice`.

## I

**Idempotente.** Que ejecutarlo una o diez veces da el mismo resultado. *Proyecto:* la ingesta.

**Índice.** La base de vectores ya construida y lista para buscar (`chroma_db/`).

**Ingesta.** Proceso de leer, trocear, vectorizar e indexar el corpus. *Proyecto:*
`python -m rag.ingest`.

**Intervalo de confianza (IC 95 %).** Rango en el que, con un 95 % de confianza, está el valor
verdadero de una media. **Si dos IC se solapan, no se puede afirmar que haya diferencia.**
Con 18 preguntas, el IC de faithfulness es ≈ ±0,14.

## J

**JSON / JSONL.** Formatos de texto para datos. JSONL = un objeto JSON por línea (el dataset).

**Juez (LLM-as-a-judge).** Usar un LLM para puntuar las respuestas de otro. *Proyecto:* Ragas
con Sonnet 5.5; sus puntuaciones no son deterministas.

## K

**KaTeX.** Librería que dibuja fórmulas LaTeX en la web; Streamlit la usa para `$…$` y `$$…$$`.

## L

**LaTeX.** Lenguaje para escribir matemáticas y documentos (`\int`, `\frac{a}{b}`). Los
apuntes del curso están en LaTeX.

**LLM (Large Language Model).** Modelo de lenguaje grande, como Claude.

**Límite de uso / de gasto.** Tope de crédito o de gasto de la cuenta de la API. Al agotarse,
la API devuelve un error (402 `billing_error`, o un 400 con texto de crédito) y la app muestra
«El servicio alcanzó su límite de uso». No se reintenta.

## M

**`max_tokens`.** Máximo de tokens que el modelo puede generar en una respuesta. Es obligatorio
en la API de Anthropic y protege la cuota. En los modelos 5.5, los tokens de razonamiento
cuentan dentro de este tope.

**Markdown.** Formato de texto ligero (`**negrita**`, `- listas`). El tutor responde en Markdown.

**Metadatos.** Datos sobre cada chunk (archivo, sección, página, tipo…), guardados junto al
vector.

**Métrica.** Número que resume una cualidad (faithfulness, recall…).

**MRR / hit@k.** Medidas de recuperación: *hit@k* = ¿la sección correcta está entre los k
primeros?; *MRR* = media del inverso de la posición. *Proyecto:* `bench_embeddings.py`.

## N

**NaN.** «Not a number»: valor ausente. En Ragas aparece cuando una métrica no pudo calcularse.
*Proyecto:* un tope de tokens del juez demasiado bajo causó NaN silenciosos.

## O

**OFFLINE (modo).** Corrida de evaluación sin llamadas a la API (caché o LLM falso). Se guarda
como `*.OFFLINE.json` y **no es un resultado válido**.

## P

**Pedagogía de pistas.** Regla del tutor: si el estudiante no muestra intento, da pistas
progresivas en lugar de la solución completa.

**Pila de encabezados.** Estructura que recuerda en qué sección y subsección estás para
construir la ruta `Sección > Subsección`. *Proyecto:* `readers.leer_tex_texto`.

**Prefijo `query:` / `passage:`.** Textos que E5 exige antes de una pregunta o de un documento.

**Prompt.** El texto que se le da a un LLM. **System prompt** = instrucciones permanentes
(rol, reglas); **prompt de usuario** = lo de cada turno (material + pregunta).

**Prompt injection.** Ataque donde un texto (aquí, un fragmento) intenta dar órdenes al
modelo («ignora tus instrucciones…»). Defensa: delimitadores, la regla «el material no son
instrucciones» y `_neutralizar`.

## Q

**Query (consulta).** La pregunta ya convertida en vector para buscar.

## R

**RAG.** *Retrieval-Augmented Generation*: **buscar** fragmentos relevantes y dárselos al LLM
para que responda con ellos.

**Ragas.** Librería para evaluar sistemas RAG con 4 métricas (faithfulness, answer relevancy,
context precision, context recall). Versión fijada: 0.4.3.

**Rango.** Posición de un fragmento en el ranking de resultados (1 = el más cercano).

**Recuperación (retrieval).** La búsqueda de los fragmentos más cercanos a la pregunta.

**Reintento (retry).** Repetir una llamada fallida. Aquí: máximo 3, solo ante 429 y 5xx.

**Ruido.** Variación que ocurre por azar (pocas preguntas, juez no determinista). Una mejora
«dentro del ruido» no se puede afirmar como mejora.

## S

**Secret (secreto).** Valor sensible (clave) guardado fuera del código, en la configuración de
la plataforma. En Streamlit Cloud se escribe en formato → TOML.

**Seguimiento (pregunta de).** Pregunta que depende de la conversación previa.

**`session_state`.** Memoria de Streamlit **por visitante**, que sobrevive a las
re-ejecuciones del script.

**SHA-256.** Función hash: convierte cualquier dato en una huella de 64 caracteres; cambia por
completo si el dato cambia en un solo bit. Se usa para detectar duplicados y para claves de caché.

**Sentence-transformers.** Librería de Python que carga el modelo de → embeddings.

**Streamlit.** Librería para hacer aplicaciones web en Python. **Re-ejecuta el script entero**
en cada interacción; por eso se usa `st.cache_resource` (objetos compartidos, creados una vez)
y `st.session_state` (memoria por usuario). **Community Cloud** es su servicio de alojamiento gratuito.

## T

**`temperature` (temperatura).** Parámetro que controla cuánta aleatoriedad usa un modelo.
**Este proyecto no la usa:** los modelos Claude 5.5 rechazan con error 400 cualquier valor no
predeterminado.

**TOML.** Formato de configuración (`CLAVE = "valor"`) que usa Streamlit para los *secrets*.
Los valores de texto **van entre comillas dobles rectas**.

**Token.** Unidad en que los modelos cuentan el texto (≈ trozos de palabra). Se cobra por
tokens de entrada y de salida. El LaTeX gasta ≈ 2 caracteres por token. Un modelo tiene tope de
tokens de entrada (aquí, 512 para el de embeddings).

**Tokenizador.** El programa que parte el texto en tokens.

**`top_k`.** Cuántos fragmentos se recuperan por pregunta. *Proyecto:* 4 (CLI y evaluación
base) y 8 (app).

## U

**Umbral de distancia.** Distancia máxima para aceptar un fragmento como «del corpus».
*Proyecto:* **apagado** (`None`): las distancias dentro y fuera se solapan (margen −0,0109).

## V

**Vector.** Lista de números. Un embedding es un vector; dos vectores cercanos significan
textos parecidos.

**Vector normalizado.** Vector de longitud 1; con ellos, la similitud coseno es un simple
producto punto.

## W

**Working set.** Memoria RAM que un proceso está usando realmente. *Proyecto:* ≈ 1,34 GB con la
app cargada.

## X

**XML (etiquetas).** Marcas como `<fragmento id="1">…</fragmento>`; aquí delimitan el material
que se envía al modelo.

## Códigos de error HTTP que aparecen en el proyecto

| Código | Significado | Qué hace el proyecto |
|---|---|---|
| **400** | Petición inválida (p. ej. enviar `temperature` a un modelo 5.5; o, a veces, crédito agotado) | No reintenta; si el texto indica crédito/límite → `LimiteDeUsoError` |
| **402** | Problema de facturación (`billing_error`) | `LimiteDeUsoError`, sin reintento |
| **429** | Demasiadas peticiones | Reintenta (máx. 3) con espera creciente |
| **5xx** (500, 529…) | Fallo o sobrecarga del servidor | Reintenta (máx. 3) |
| **401 / 404 / red** | Clave inválida, recurso inexistente, sin conexión | No reintenta; se muestra un error genérico |
