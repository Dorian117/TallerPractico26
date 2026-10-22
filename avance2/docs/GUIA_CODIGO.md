# Guía del código: qué hace cada parte y por qué

Esta guía está escrita para **entender** el proyecto, no solo para usarlo. Si una palabra
técnica no te suena, búscala en [`GLOSARIO.md`](GLOSARIO.md). Las decisiones con sus cifras
reales están en [`DECISIONES.md`](DECISIONES.md); aquí se explica el **código**.

---

## 1. La idea completa en un minuto

Un LLM (Claude) sabe mucho, pero **no conoce tus apuntes** y a veces inventa. RAG
(*Retrieval-Augmented Generation*) lo resuelve en dos tiempos:

1. **Buscar** en tus apuntes los fragmentos más parecidos a la pregunta (sin usar el LLM).
2. **Dárselos al LLM** junto con la pregunta y obligarlo a responder con ellos y a citarlos.

El proyecto tiene **tres actividades distintas**, y conviene no mezclarlas:

```
A) INGESTA (se hace una vez, o al cambiar los apuntes; todo en tu máquina)
   corpus/*.txt ──► readers ──► chunking ──► embeddings ──► store (Chroma)
   "leer"          "limpiar y    "trocear"    "convertir     "guardar el
                    separar por               en vectores"   índice en
                    secciones"                               chroma_db/"

B) CONSULTA (cada pregunta del estudiante)
   pregunta ─► [generate.condensar] ─► [retrieve] ─► [prompts.armar_mensajes] ─► [llm] ─► respuesta
               solo si hay historial   top_k          system + ejemplos +         Claude    + fuentes
                                       fragmentos     historial + fragmentos
                                                      + pregunta
C) EVALUACIÓN (para medir la calidad; no la usa el estudiante)
   dataset.jsonl ─► run_eval ─► (pipeline B por cada pregunta) ─► Ragas (juez LLM) ─► tablas/gráficos
```

**Qué sale de tu máquina:** solo en (B), y solo hacia el proveedor del LLM: pregunta,
historial, los `top_k` fragmentos, y el texto fijo del prompt. Los embeddings y el índice
nunca salen.

---

## 2. Mapa de carpetas

```
avance2/
├── app.py                 Chat web (Streamlit). Es la "cara" del proyecto.
├── rag/                   El motor. Cada archivo hace UNA cosa.
│   ├── config.py          Todos los parámetros en un solo sitio
│   ├── readers.py         Leer archivos (LaTeX, PDF, Word, MD, HTML)
│   ├── chunking.py        Trocear en fragmentos sin romper fórmulas
│   ├── embeddings.py      Texto → vector (modelo local)
│   ├── store.py           Guardar/consultar el índice (Chroma)
│   ├── ingest.py          Orquesta A) la ingesta
│   ├── retrieve.py        Buscar los top_k fragmentos
│   ├── llm.py             Hablar con Claude (reintentos, caché, límites)
│   ├── prompts.py         Los textos que se le dan al LLM
│   ├── generate.py        Orquesta B) la consulta completa
│   └── cli.py             Tutor por consola
├── eval/                  C) Evaluación con Ragas y análisis
├── tests/                 Pruebas que NO gastan API
├── corpus/                Tus apuntes (6 archivos .txt con LaTeX)
├── docs/                  Documentación
├── .streamlit/            Configuración de Streamlit
├── chroma_db/             El índice (se genera; no va a git)
└── .cache_llm/            Caché de respuestas del LLM (se genera; no va a git)
```

**Regla de dependencias** (quién importa a quién): `config` no importa nada del proyecto;
`readers` → `chunking` → `ingest`; `embeddings` y `store` los usan `ingest` y `retrieve`;
`prompts` usa `retrieve.Fragmento`; `generate` junta `retrieve`, `prompts` y `llm`;
`app.py` y `cli.py` solo llaman a `generate.responder`. Nada del motor importa `app.py`.

---

## 3. El recorrido de una pregunta, paso a paso

Supón que el estudiante escribe, en el **segundo** turno: *«¿Y si fuera 9x²?»*.

| # | Dónde | Qué pasa |
|---|---|---|
| 1 | `app.py` | `st.chat_input` recibe el texto. Se comprueba el límite de preguntas y se llama a `responder(pregunta, historial, top_k=8, llm=...)` |
| 2 | `generate.condensar_pregunta` | Hay historial, así que se pide al LLM que reescriba la pregunta **autónoma** («¿Cómo resuelvo ∫dx/√(9x²+1)…?»). Sin historial, este paso se salta (ahorra una llamada) |
| 3 | `retrieve.recuperar` | Convierte la pregunta condensada en un vector (`embeddings.embeber_consulta`, con prefijo `query: `), le pide a Chroma los 8 vectores más cercanos y devuelve 8 `Fragmento` con su distancia |
| 4 | `prompts.armar_mensajes` | Construye la lista de mensajes: **ejemplos few-shot + historial real + turno actual** (que lleva los 8 fragmentos dentro de `<material_curso>`). La pregunta que va aquí es la **original**, no la condensada |
| 5 | `llm.LLMAnthropic.generar` | Envía `system=SYSTEM_PROMPT`, los mensajes y `max_tokens=8000` a Claude. Reintenta solo ante 429/5xx (máx. 3) |
| 6 | `generate.responder` | Revisa si se cortó por `max_tokens`, si el modelo rechazó, si vino vacío; calcula las citas inválidas; devuelve un `Respuesta` |
| 7 | `app.py` | Pinta la respuesta (LaTeX con KaTeX), el expander «Fuentes» y los contadores; guarda el turno en `st.session_state` |

**Por qué se busca con la pregunta condensada pero se genera con la original:** la
recuperación necesita una pregunta que se entienda sola (el buscador no ve el historial);
la generación sí ve el historial real, así que conserva el tono y los detalles de lo que el
estudiante escribió.

---

## 4. Módulo por módulo

### 4.1 `rag/config.py` — los mandos de control

**Qué es:** un archivo de constantes. Cualquier número que se pueda afinar vive aquí y en
ningún otro sitio. Así, cuando en la Fase 2 se cambió `top_k`, fue **una línea**.

**Lo que hace al importarse (orden importante):**
1. Pone `ANONYMIZED_TELEMETRY=False` y `RAGAS_DO_NOT_TRACK=true` **antes** de importar Chroma
   o Ragas. Con `setdefault`, si ya estaban definidas en el entorno, se respetan.
2. `load_dotenv()` lee `avance2/.env` (la clave). Nunca se imprime.
3. Define rutas relativas a la carpeta del proyecto (`RAIZ`), para que sea autocontenido.

**Grupos de parámetros:**

| Grupo | Constantes | Para qué |
|---|---|---|
| Chunking | `CHUNK_TAMANO=1800`, `CHUNK_SOLAPE=200`, `CHUNK_MINIMO=200` | Tamaño objetivo (en caracteres), cola repetida entre chunks y mínimo para no crear trozos diminutos |
| Embeddings | `MODELO_EMBEDDINGS`, `PREFIJO_CONSULTA/PASAJE`, `EMBEDDING_MAX_TOKENS=512`, `EMBEDDING_TEXTO_MODO="limpio"` | Qué modelo, qué prefijos exige E5, tope de tokens y qué versión del texto se vectoriza |
| Recuperación | `TOP_K=4`, `UMBRAL_DISTANCIA=None` | Cuántos fragmentos; el umbral está apagado a propósito |
| LLM | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_JUDGE_MODEL`, `MAX_TOKENS_*`, `REINTENTOS_API=3`, `CACHE_LLM_DIR` | Qué modelo y cuántos tokens como máximo en cada llamada |
| Historial | `HISTORIAL_MAX_TURNOS=6`, `CONDENSACION_TURNOS=4`, `CONDENSACION_RECORTE=600` | Cuánta conversación se reenvía (más historial = más tokens = más coste) |
| App | `APP_TOP_K=8`, `APP_MAX_PREGUNTAS=15`, `APP_MAX_CARACTERES=1500`, `APP_FUENTE_VISTA=700` | Límites de la app pública |

**Nota importante:** **no existe ningún parámetro de temperatura.** Los modelos Claude 5.5
rechazan con error 400 cualquier valor distinto del suyo por defecto, así que se decidió
quitar la opción por completo.

`TOP_K=4` (CLI y evaluación base) y `APP_TOP_K=8` (app) son valores distintos a propósito.

---

### 4.2 `rag/readers.py` — leer cada formato

**Problema que resuelve:** cada formato guarda la estructura de forma distinta (`\section`
en LaTeX, «Heading 1» en Word, `#` en Markdown, `<h1>` en HTML, tamaño de letra en PDF).
Si el lector entrega **bloques ya agrupados por sección**, el chunker puede cortar
respetando esa estructura.

**Estructuras de datos:**
- `Bloque(texto, seccion, pagina)`: un trozo de texto con su ruta de encabezados
  (`"Ejemplos > Ejemplo 1: ..."`) y página (0 = el formato no tiene páginas).
- `DocumentoLeido(archivo, titulo, formato, sha256, bloques, advertencias)`: lo que devuelve
  cualquier lector. El resto del proyecto **no sabe** de qué formato vino el texto.

**El lector de LaTeX (el que usa tu corpus):**
1. `leer_txt` detecta con `es_latex` si el `.txt` en realidad es LaTeX (mira si contiene
   `\documentclass` o `\begin{document}`), porque la extensión no es de fiar.
2. `leer_tex_texto`: saca el título (`_titulo_tex`), se queda con el texto entre
   `\begin{document}` y `\end{document}`, descarta el membrete institucional y **parte por
   `\section`, `\subsection`, `\subsubsection`** manteniendo una **pila de encabezados**:
   la «sección» de un bloque es la ruta completa (`"Casos > Caso 1"`).
3. `limpiar_tex` limpia cada tramo **sin tocar las fórmulas**. Truco central: separa primero
   las fórmulas protegidas (`$$…$$`, `\[…\]`, `align`, `equation`, etc.; ver `_RE_PROTEGIDO`)
   y aplica la limpieza solo al texto de fuera (`_limpiar_fuera_de_mate`).

**Qué limpia `_limpiar_fuera_de_mate`:**

| Entrada LaTeX | Resultado |
|---|---|
| `\begin{tikzpicture}…\node{texto}…` | `Diagrama:` + el texto de cada nodo (`_nodos_tikz`) |
| `\begin{tabular}…` | filas `\| a \| b \|` (`_tabla_a_texto`) |
| `center`, `multicols`, `tcolorbox`… | se quitan las marcas, se conserva el contenido |
| `\item` | `- ` (lista Markdown) |
| `\textbf{X}`, `\textit{X}`, `\emph{X}` | `**X**`, `*X*` |
| `\vspace`, `\newpage`, `\small`… | se eliminan |
| `\\` | salto de línea |
| comentarios `%…` | se eliminan |

`_cierre_llave` / `_argumento` / `_reemplazar_comando` existen porque las expresiones
regulares no saben contar llaves anidadas: `\textbf{a \frac{1}{2} b}` exige encontrar la
llave que **cierra** de verdad.

**Otros lectores:**
- `leer_md`: encabezados `#`; ignora `#` dentro de bloques de código.
- `leer_html`: quita `script/style/nav/footer`, usa `h1…h6` como secciones (BeautifulSoup).
- `leer_docx`: estilos «Heading n»; rescata el texto lineal de las ecuaciones de Word (OMML)
  y avisa de que **no es LaTeX**.
- `leer_pdf` (pymupdf): infiere encabezados por **tamaño de fuente** (≥ 1,2 × la mediana y
  ≤ 100 caracteres). `_calidad_pdf` avisa si las fórmulas salieron deformadas. Por eso el
  proyecto abandonó los PDF y usa los fuentes LaTeX.
- `LECTORES` es un diccionario extensión → función; `leer_documento` lo consulta.

> `pypdf` está en `requirements.txt` pero **ningún módulo lo importa** (se dejó como
> alternativa de lectura); el PDF se lee con `pymupdf`.

**`sha256`:** huella del archivo; sirve para detectar duplicados en la ingesta y para la
«huella» del índice.

---

### 4.3 `rag/chunking.py` — trocear sin romper nada

**Por qué trocear:** el LLM no recibe 60 KB de apuntes (caro y poco preciso), y el modelo de
embeddings tiene un tope de 512 tokens. Se parte cada bloque en **chunks** (fragmentos).

**Cinco principios** (cada uno corrige un fallo concreto en RAG matemático):
1. **Respetar la estructura:** si un bloque cabe en `CHUNK_TAMANO`, es **un** chunk.
2. **Jamás cortar una fórmula.** Una mitad de `\begin{align*}` no es LaTeX válido.
3. **Cortes jerárquicos:** primero entre párrafos, luego entre líneas, solo al final entre oraciones.
4. **Solape con piezas completas:** la cola que se repite en el siguiente chunk nunca parte una fórmula.
5. **Sin trozos diminutos:** un bloque < `CHUNK_MINIMO` se fusiona con el siguiente.

**Cómo se protege una fórmula (átomos):** `_enmascarar` sustituye cada fórmula
(`$$…$$`, `\[…\]`, `\begin…\end`, `$…$`) por una marca opaca (`\x00 N \x00`). Así ningún
salto de línea o punto **dentro** de una fórmula puede provocar un corte. `_restaurar` las
devuelve. `_partir` corta por la jerarquía de `_SEPARADORES`; una pieza que sigue siendo
demasiado grande (una tabla gigante) se deja **entera**: «mejor entera que rota».

**`trocear_texto`:** junta piezas hasta pasarse de `tam`, cierra el chunk y arranca el
siguiente con una cola de piezas completas de unos `solape` caracteres.

**El detalle más importante: medir en tokens, no en caracteres.**
`_trocear_bloque_en_tokens` trocea por caracteres y **comprueba cuántos tokens** ocuparía
cada trozo con el tokenizador del modelo de embeddings. Si alguno supera 512, repite con un
tamaño un 20 % menor (`tam × 0.8`) hasta que quepa (o `tam < 300`: entonces se acepta y se
marca `excede_tokens=True`). Razón: el LaTeX tokeniza muy denso (≈ 2 caracteres por token),
así que 1800 caracteres pueden ser ~800 tokens, y el modelo **truncaría el final en
silencio**, perdiendo la parte que quizá contenía la respuesta.

**`Chunk` tiene dos textos distintos:**
- `texto`: el LaTeX completo. Es lo que se guarda, se cita y se envía al LLM.
- `texto_embedding`: la versión que se convierte en vector. `_texto_embedding` le antepone
  `Título del documento > Sección` (porque «Ejemplo 2» solo no dice de qué tema es) y
  `normalizar_para_embedding` quita ruido de maquetación (`\displaystyle`, `&`, `\\`…).
  Modos: `crudo` (sin tocar), `limpio` (por defecto) y `verbal` (`\int` → «integral»).

**`clasificar_tipo`:** etiqueta cada chunk (`teoria`, `ejemplo`, `ejercicio`, `respuesta`,
`procedimiento`, `formulario`, `definicion`) mirando el **último encabezado** de la ruta con
expresiones regulares. Es un metadato filtrable; hoy no se usa para filtrar.

**`chunk_id`:** `<nombre-del-archivo>-<n>` (por ejemplo `c-digo-latex-de-sustitucion-007`).

---

### 4.4 `rag/embeddings.py` — texto a números

**Qué es un embedding:** un vector (aquí de 384 números) que representa el *significado* de un
texto; textos parecidos quedan cerca. Se usa el modelo `intfloat/multilingual-e5-small`,
**local**: tras descargar los pesos una vez (471 MB), nada sale de la máquina.

- `cargar_modelo()` (con `lru_cache`: se carga una sola vez por proceso). Intenta primero
  `local_files_only=True` (sin red); solo si no está descargado, contacta con Hugging Face.
  Fija `max_seq_length = 512` explícitamente.
- `contar_tokens(texto)`: lo usa el chunker.
- `_embeber`: `normalize_embeddings=True` → vectores de norma 1. Así la similitud coseno es
  un producto punto y la **distancia** coseno de Chroma (1 − similitud) queda en [0, 2], con
  0 = idéntico.
- **Prefijos de E5:** `embeber_pasajes` antepone `passage: ` a los documentos y
  `embeber_consulta(s)` antepone `query: ` a las preguntas. El modelo se entrenó así;
  mezclarlos o quitarlos degrada la búsqueda.

`import torch` tarda unos segundos, por eso los `import` están **dentro** de las funciones:
los módulos que solo leen o trocean texto no pagan ese coste.

---

### 4.5 `rag/store.py` — el índice en Chroma

Chroma es una base de datos de vectores que vive en una carpeta (`chroma_db/`; SQLite +
ficheros del índice HNSW). Sin servidor.

- `_cliente()`: `PersistentClient` con telemetría desactivada.
- `reconstruir(chunks, vectores, huella)`: **borra y recrea** la colección. Por eso la
  ingesta es **idempotente**: ejecutarla 1 o 10 veces deja el mismo índice, sin duplicados ni
  restos de archivos que ya no existen. La colección se crea con
  `metadata={"hnsw:space": "cosine", "huella": ...}` y `embedding_function=None`
  (los vectores los calculamos nosotros, con E5; así Chroma no descarga otro modelo por su cuenta).
  Por cada chunk guarda: `id`, vector, `texto` y metadatos (`fuente`, `documento`, `seccion`,
  `pagina`, `tipo`, `formato`, `n_chunk`, `excede_tokens`).
- `huella(...)`: hash de los archivos del corpus **más** los parámetros que afectan al índice
  (modelo, modo, tamaños). Si algo cambia, la huella cambia.
- `indice_vigente(huella)`: ¿la huella guardada coincide? La usa la app para decidir si hay
  que reconstruir.

---

### 4.6 `rag/ingest.py` — el orquestador de la ingesta

`python -m rag.ingest` hace, en orden: `descubrir` (archivos con extensión soportada,
ignorando los que empiezan por punto) → `leer_corpus` → `construir_chunks` → `informe` →
embeber → `store.reconstruir`.

Detalles que importan:
- **Un archivo ilegible no detiene la ingesta:** se anota en avisos y se sigue.
- **Duplicados** (mismo `sha256`): se omiten con aviso; si no, el mismo fragmento saldría
  varias veces en el `top_k`.
- `informe` imprime documentos, secciones, nº de chunks, mín/mediana/máx de caracteres y
  **tokens**, y alerta de chunks indivisibles que superan el tope.
- Opciones: `--muestras N`, `--sin-indexar`, `--modo`, `--corpus`.
- **`asegurar_indice()`** (la usa la app): lee el corpus, calcula la huella; si el índice
  existe y coincide, devuelve `"vigente"`; si no, ejecuta la ingesta completa y devuelve
  `"reconstruido"`. Gracias a esto, un despliegue nuevo construye su índice solo en el primer arranque.

---

### 4.7 `rag/retrieve.py` — buscar

`recuperar(pregunta, top_k, umbral)`:
1. `embeddings.embeber_consulta(pregunta)` → vector.
2. `store.obtener().query(...)` → los `top_k` más cercanos, con texto, metadatos y distancias.
3. Los empaqueta como `Fragmento(texto, fuente, documento, seccion, pagina, tipo, chunk_id, distancia, rango)`.
4. Si hay `umbral`, descarta los que están más lejos. **Hoy `UMBRAL_DISTANCIA` es `None`:** la
   evaluación mostró que las distancias dentro y fuera del corpus se **solapan**, y el LLM ya
   se abstiene bien por sí solo.

`python -m rag.retrieve "pregunta"` imprime los fragmentos y sus distancias **sin llamar al
LLM** (gratis): la mejor herramienta para depurar «¿por qué no encontró esto?».

---

### 4.8 `rag/llm.py` — hablar con Claude (y no arruinarse)

Todo el proyecto llama a **`llm.generar(system=..., mensajes=..., max_tokens=...)`** y recibe
una `RespuestaLLM(texto, modelo, tokens_entrada, tokens_salida, detenido_por, desde_cache)`.
Qué proveedor hay detrás es un detalle de este archivo.

**Las 5 reglas** (están en la cabecera del archivo y las verifican los tests):
1. **Sin temperatura** (`temperature`, `top_p`, `top_k` jamás se envían).
2. **`max_tokens` explícito** en cada llamada.
3. **Reintentos:** máximo 3, con espera exponencial, **solo** ante 429 (demasiadas
   peticiones) y 5xx (error del servidor). El cliente se crea con `max_retries=0` para que el
   único bucle sea el de `_con_reintentos`.
4. **`system` en su parámetro**, no dentro de los mensajes; los mensajes alternan
   `user`/`assistant` empezando por `user`.
5. **Caché en disco** por hash de (modelo + system + mensajes + max_tokens).

**Piezas:**
- `LimiteDeUsoError`: se lanza cuando se agota el crédito o el límite de gasto
  (`es_limite_de_uso`: HTTP 402, tipo `billing_error`, o un 400 con palabras como
  «credit balance» o «usage limit»). **No se reintenta**: reintentar no lo arregla. Su mensaje
  es fijo: «El servicio alcanzó su límite de uso».
  *(Los textos exactos del 400 por crédito no están verificados con un caso real.)*
- `_con_reintentos(llamada, clasificar)` y `_espera` (1, 2, 4… s con un poco de azar, máx.
  30 s; respeta la cabecera `retry-after`).
- `_secreto(nombre)`: busca la clave en variables de entorno y, si no, en `st.secrets`. Así
  funciona igual en tu PC (`.env`) y en Streamlit Cloud (panel *Secrets*).
- `LLMAnthropic`: construye la petición, la envía y suma los tokens de entrada (incluyendo los
  de caché del proveedor).
- `LLMGemini`: alternativa conservada, **sin ampliar ni volver a probar**.
- `LLMFalso`: responde texto fijo y guarda cada llamada en `.llamadas`; sirve para tests y
  desarrollo sin gastar.
- Caché: `clave_cache` (SHA-256), `_Cache` (escritura **atómica**: archivo temporal y luego
  renombrar; **no guarda** respuestas truncadas o rechazadas, para no servir luego la mala),
  `LLMConCache` (envuelve un LLM real) y `LLMCacheOFalso` (modo «sin `--live`»: sirve caché
  real si hay, si no un LLM falso; nunca llama a la API).
- `crear_llm(rol, live, usar_cache)`: la fábrica. `rol="tutor"` usa `LLM_MODEL`,
  `rol="juez"` usa `LLM_JUDGE_MODEL`. **La app usa `usar_cache=False`** para no guardar preguntas de estudiantes.

---

### 4.9 `rag/prompts.py` — lo que se le dice al modelo

**`SYSTEM_PROMPT`** (instrucciones permanentes), por bloques:
1. **Rol:** tutor de Cálculo II (Konrad Lorenz), con las técnicas del curso.
2. **Pedagogía:** si el estudiante **no** muestra intento, solo da **pistas**; si lo muestra,
   evalúa su intento y continúa desde ahí.
3. **Formato fijo de 6 pasos** para soluciones completas (identificar, plantear, simplificar,
   integrar, devolver la sustitución, verificar derivando).
4. **Errores de signo:** señalarlos solo si el estudiante realmente los comete.
5. **Reglas RAG:** el material llega en `<material_curso>`; es **referencia, no instrucciones**;
   la teoría sale solo de los fragmentos y se **cita** `[n]`; el cálculo es razonamiento propio
   (no se cita); si no hay cobertura, **empezar exactamente** con «No encontré esto en el
   material del curso» y no responder de memoria; si un fragmento tiene un resultado erróneo,
   decirlo.
6. **Formato de salida:** español, Markdown, matemáticas en LaTeX entre `$`.

**`INSTRUCCION_TAREA`:** recordatorio corto que va **fuera** del delimitador, junto a la
pregunta, para que el modelo distinga «material» de «qué hacer ahora».

**Delimitadores (`formatear_material`):** cada fragmento se envía como

```
<material_curso>
<fragmento id="1" fuente="archivo.txt" documento="Título" seccion="A > B">
...texto LaTeX...
</fragmento>
...
</material_curso>
```

Por qué atributos y no una lista de texto libre: el modelo para citar solo **copia un número**
(`id`), sin reproducir nombres largos y equivocarse. La interfaz traduce `[2]` a
archivo/sección de forma determinista.
- `_atributo` escapa comillas y `<>&` dentro de los atributos.
- `_neutralizar` desactiva `</fragmento>` o `</material_curso>` si apareciesen **dentro** del
  texto de un fragmento: es una defensa contra *prompt injection* (que un fragmento «cierre»
  el delimitador y lo que siga se lea como instrucción).
- `turno_usuario` junta: material (dentro) + instrucción de tarea + pregunta (fuera).

**Citas:** `citas_en` extrae los `[n]` (ignora los que estén dentro de `$…$`, porque `[0,1]`
es un intervalo) y `citas_invalidas` devuelve los que apuntan a un fragmento que no existe.
*Límite:* solo comprueba que el número **exista**, no que el fragmento **respalde** lo dicho.

**Few-shot (`EJEMPLOS_FEW_SHOT`):** 4 ejemplos (10 mensajes) que enseñan al modelo el formato
esperado: (A) sustitución trigonométrica con intento previo, (B) sustitución más partes,
(C) rechazo fuera de corpus, (D) pistas más un seguimiento con error de signo. Sus fragmentos
son **copias del corpus real** y las matemáticas se verificaron numéricamente. Se envían en
**todas** las llamadas de generación: es una parte importante del coste en tokens.

**`armar_mensajes`:** `EJEMPLOS_FEW_SHOT + historial + [turno actual]`.

**`PROMPT_CONDENSACION`:** plantilla para reescribir un seguimiento como pregunta autónoma
(conserva el LaTeX, aplica el cambio de dato, devuelve una sola línea).

---

### 4.10 `rag/generate.py` — el director de orquesta

- `Turno(rol, texto)` y `Respuesta(texto, fragmentos, pregunta_recuperacion, citas_invalidas,
  llamo_modelo, modelo, tokens_*, segundos, desde_cache, avisos)`.
- `condensar_pregunta`: sin historial devuelve la pregunta tal cual (sin llamada). Con
  historial, manda al LLM los últimos 4 turnos recortados a 600 caracteres y toma la
  **primera línea** de la respuesta.
- `_historial_a_mensajes`: toma los últimos 6 turnos reales y garantiza que empiecen por
  `user`.
- **`responder(pregunta, historial, top_k, llm)`:**
  1. condensar (si falla, busca con la pregunta original y deja un aviso; un
     `LimiteDeUsoError` sí se propaga);
  2. `recuperar`;
  3. (gancho del umbral; inactivo);
  4. `armar_mensajes` + `llm.generar`;
  5. avisos si se cortó por `max_tokens`, hubo rechazo o la respuesta vino vacía;
  6. devuelve `Respuesta`.
- Si `llm` no se pasa, crea uno **sin caché** (`usar_cache=False`).

---

### 4.11 `rag/cli.py` — el tutor por consola

`python -m rag.cli` abre un bucle (`nueva` reinicia, `salir` termina). `--demo` ejecuta la
conversación guionizada de 3 turnos (solución completa → seguimiento → fuera de corpus) con
caché. **Un turno que falla no se guarda en el historial**, para no contaminar la
conversación. Muestra modelo, tokens, tiempo, si vino de caché y citas inválidas.

---

### 4.12 `app.py` — la app web

Streamlit **vuelve a ejecutar todo el archivo de arriba abajo cada vez que el usuario hace
algo**. Por eso hay dos mecanismos para no repetir trabajo caro:

- **`@st.cache_resource`:** objetos que se crean **una vez por proceso** y se comparten entre
  sesiones: `preparar_indice()` (asegura el índice y precarga el modelo de embeddings) y
  `obtener_llm()`.
- **`st.session_state`:** memoria **por visitante**: `mensajes`, `preguntas`,
  `tokens_entrada`, `tokens_salida`. `reiniciar()` la vacía («Nueva conversación»).

**Orden del archivo:** `set_page_config` (debe ser lo primero de Streamlit) → helpers
(`md`, `historial`, `fuentes_de`, `mostrar_fuentes`) → barra lateral → título → comprobar la
clave (si falta: aviso y `st.stop()`) → preparar índice → pintar saludo e historial →
`chat_input` → si hay pregunta, responder.

**Piezas:**
- `md(texto)`: convierte un salto de línea simple en salto duro (dos espacios + `\n`), porque
  Markdown junta las líneas sueltas. Los párrafos no cambian.
- `fuentes_de(r)`: convierte los `Fragmento` en diccionarios simples (con los primeros 700
  caracteres) para guardarlos en `session_state`.
- **Límites:** `APP_MAX_PREGUNTAS` (el `chat_input` queda `disabled`) y
  `max_chars=APP_MAX_CARACTERES`.
- **Errores:** `LimiteDeUsoError` → mensaje fijo. Cualquier otro → mensaje genérico; el
  **nombre** del error va al log del servidor, nunca el detalle al usuario (podría filtrar rutas
  o texto de la API). Una pregunta fallida no cuenta ni se guarda.
- Al terminar guarda los dos mensajes, suma contadores y hace `st.rerun()` para refrescar
  la barra lateral.

**Configuración de Streamlit (`.streamlit/config.toml`):** `gatherUsageStats=false`
(sin telemetría), `headless=true` (no abre el navegador), `fileWatcherType="none"`
(el vigilante de archivos recorría `transformers` y llenaba el log de errores de
`torchvision`), `toolbarMode="minimal"`.

---

### 4.12b Cuánto cuesta una pregunta y de dónde salen los tokens

En la prueba real, cada respuesta usó ≈ 13 000–15 000 tokens de **entrada** y 500–2 000 de
salida. La entrada se compone de: system prompt + 10 mensajes few-shot + el historial +
**8 fragmentos** + la pregunta. Es lo que sube el coste al subir `top_k` o al alargar la
conversación. El contador de la barra lateral suma los tokens de **generación** (no los de la
llamada de condensación).

---

### 4.13 `eval/` — medir la calidad

**El dataset (`dataset.jsonl`):** 23 preguntas, una por línea. Campos: `question`,
`ground_truth` (respuesta de referencia), `tipo` (`definicion`, `procedimiento`, `ejercicio`,
`seguimiento`, `fuera_de_corpus`), `fuente_esperada`, `seccion_esperada` y, en los
seguimientos, `historial`. 18 son del corpus y 5 de fuera.

**`run_eval.py` — qué hace una corrida:**
1. `cargar_dataset` valida los campos.
2. `generar`: para cada pregunta ejecuta el pipeline real (`responder`) y guarda respuesta,
   contextos, fuentes recuperadas, citas inválidas, tokens y si se abstuvo.
3. `metricas_abstencion` (**métricas propias**, no de Ragas): tasa de abstención correcta,
   de alucinación (fuera de corpus que **no** se abstuvo) y de abstención indebida (del corpus
   que se abstuvo). `se_abstiene` busca la frase «no encontre esto en el material del curso»
   (sin tildes ni mayúsculas) en los primeros 250 caracteres.
4. `evaluar_con_ragas`: las **4 métricas** sobre las preguntas del corpus (a las de fuera no
   se les aplican: no tienen referencia que comparar). En los seguimientos usa la pregunta
   condensada.
5. `guardar`: `.json` y `.csv` en `eval/results/`.

**Modos de gasto:** sin `--live` no se llama a ninguna API (caché real o LLM falso) y el
archivo se guarda como `.OFFLINE.json` («no es un resultado válido»). Con `--live`, todo pasa
por la caché. `--juez-final` usa el modelo más caro como juez. `--limite N` y `--sin-ragas`.
Ante **cualquier** error la corrida se detiene y lo explica; códigos de salida: 0 ok, 1
entrada inválida, 2 error, 3 límite de uso.

**`juez.py` — el juez de Ragas con Anthropic.** Ragas trae su forma de usar Anthropic, pero
**no funciona** con estos modelos (fuerza `tool_choice` y envía temperatura, y ambos dan 400).
Solución: usar el envoltorio de LangChain, con:
- `ChatAnthropicJuez` ignora cualquier asignación a `temperature/top_p/top_k` (`__setattr__`)
  y las borra del payload; cachea en disco (`_clave` **no** incluye `max_tokens`).
- `run_config()`: reintentos alineados con la regla (solo 429/5xx, `max_workers=2`).
- `JuezRagas` anota un límite de gasto, porque Ragas con `raise_exceptions=False` lo
  convertiría en `NaN` silencioso.
- `EmbeddingsLocales`: `answer_relevancy` usa los embeddings E5 locales (nada sale).
- **Hallazgo real:** con `MAX_TOKENS_JUEZ=4096` el juez se quedaba sin tokens y Ragas
  devolvía `NaN` en faithfulness para las 5 respuestas largas; se subió a 16 000.

**Las 4 métricas de Ragas** (todas puntúan de 0 a 1 usando un LLM como juez):

| Métrica | Pregunta que contesta |
|---|---|
| `faithfulness` | ¿Lo que afirma la respuesta está respaldado por los fragmentos recuperados? |
| `answer_relevancy` | ¿La respuesta es pertinente a la pregunta? |
| `context_precision` | ¿Los fragmentos relevantes aparecen **arriba** del ranking? |
| `context_recall` | ¿Los fragmentos recuperados contienen lo necesario para la respuesta de referencia? |

**`analisis.py`:** tabla Markdown y LaTeX, gráfico PNG, **intervalo de confianza por
bootstrap** (con ~18 preguntas es ancho: así se ve cuánto ruido hay), evidencia de las peores
preguntas, medias por tipo y la **calibración del umbral** (distancia del mejor fragmento
dentro vs fuera del corpus, con AUC y margen).

**`compare.py`:** compara dos corridas (antes → después), marca lo que **empeoró** y si la
diferencia cae dentro del ruido. **`bench_retrieval.py`** y **`bench_embeddings.py`** miden
**gratis** (sin API) qué configuración de recuperación o de modelo funciona mejor.

---

### 4.14 `tests/` — pruebas que no gastan

| Archivo | Qué comprueba |
|---|---|
| `smoke_readers.py` | Los lectores y el chunker (con archivos sintéticos para PDF/DOCX/HTML/MD) y que ningún chunk parte una fórmula |
| `test_llm.py` | Parámetros enviados, que no haya temperatura en el código, reintentos (429/5xx, máx. 3), que otros errores no se reintenten, límite de uso, caché, modo sin `--live`, `responder` con LLM falso. (`--live` añade **una** llamada real mínima) |
| `test_eval.py` | El juez nunca envía muestreo, `run_config`, caché del juez, detección de abstención, corrida offline, mensaje claro al detenerse |
| `test_app.py` | La app con `AppTest` y LLM falso: carga, un turno con fuentes, seguimiento, límite de preguntas, reinicio, límite de uso, errores sin filtrar detalles, falta de clave, saltos de línea |

Se ejecutan con `python -m tests.<nombre>`.

---

### 4.15 Archivos de configuración

| Archivo | Para qué | ¿Va a git? |
|---|---|---|
| `.env` | Tu clave y modelos (local) | **No** |
| `.env.example` | Plantilla sin secretos | Sí |
| `.streamlit/secrets.toml` | Clave para ejecutar con `st.secrets` en local | **No** |
| `.streamlit/secrets.toml.example` | Plantilla | Sí |
| `.streamlit/config.toml` | Config de Streamlit | Sí |
| `requirements.txt` | Dependencias de ejecución, versiones **fijadas** (`==`) para reproducir | Sí |
| `requirements-eval.txt` | Lo anterior + Ragas y análisis | Sí |
| `.gitignore` | Excluye `.env`, `.venv`, `chroma_db/`, `.cache_llm/`, resultados offline | Sí |

---

## 5. Límites conocidos (lo que el código no garantiza)

- **Las citas solo se validan por existencia**, no por respaldo: el modelo puede citar `[2]`
  para algo que el fragmento 2 no dice.
- **`faithfulness` ≈ 0,55 en la línea base:** parte es recuperación (consultas largas de
  «estudiante que ya intentó» no recuperan la sección correcta) y parte es que el tutor
  **calcula** y ese cálculo no está en los fragmentos (techo ≈ 0,64).
- **18 preguntas son pocas:** ninguna diferencia entre `top_k` 4 y 8 es distinguible del ruido.
- **El juez es un LLM:** sus puntuaciones son no deterministas; por eso la caché del juez
  fija lo ya puntuado.
- **El corpus tiene 2 respuestas erróneas conocidas** (partes e) e i)) y un archivo truncado;
  no se corrigieron por decisión tuya.
- **El umbral de distancia no se usa:** los datos no lo justifican.
- **Gemini** quedó como código alternativo sin probar tras el cambio a Anthropic.

---

## 6. Ejercicios para entenderlo de verdad

Cada uno es barato y te enseña algo concreto.

1. **Ver el buscador sin gastar:** `python -m rag.retrieve "¿qué es ILATE?"` y mira
   distancias y secciones. Prueba una pregunta fuera del temario y compara.
2. **Ver los chunks:** `python -m rag.ingest --sin-indexar --muestras 5`. ¿Alguna fórmula
   aparece cortada? (No debería.)
3. **Cambiar un parámetro:** en `config.py` pon `CHUNK_TAMANO = 900`, ejecuta
   `python -m rag.ingest` y fíjate en cuántos chunks salen y en el cambio de tokens.
   **Después devuélvelo a 1800 y vuelve a ejecutar `python -m rag.ingest`**: la ingesta
   reconstruye el índice completo, así que si no la repites, la app seguirá usando el índice
   de prueba (aunque la app lo detectaría sola al arrancar, porque la huella cambia).
4. **Entender la condensación:** `python -m rag.cli --demo` (usa caché, casi sin gasto) y
   observa la línea «consulta usada para buscar» en el turno 2.
5. **Romper algo a propósito:** quita la clave de `.env` y arranca la app: verás el aviso y
   que no aparece el chat.
6. **Leer un test:** abre `tests/test_llm.py::test_reintenta_429_y_5xx_con_maximo_3`; es la
   forma más corta de entender la regla de reintentos.
