"""Configuracion central del pipeline RAG.

Todos los parametros que se pueden afinar (tamano de chunk, top_k, modelo de
embeddings, umbral...) viven AQUI y en ningun otro sitio. Asi, cuando en la
Fase 2 se itere sobre uno de ellos, el cambio es de una linea y queda claro
que se modifico solo eso.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 1. Privacidad: se fijan las banderas de telemetria ANTES de importar
#    chromadb o ragas. Con setdefault, un valor ya definido en el entorno
#    manda, y el .env (load_dotenv no pisa lo existente) tampoco puede
#    reactivarlas por accidente.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# 2. Cargar .env. python-dotenv busca hacia arriba desde este archivo:
#    primero avance2/.env y, si no existe, el .env de la raiz del repo (el
#    que ya usa el Avance 1). Solo se lee la clave; nunca se imprime.
load_dotenv()

# 3. Rutas. Todo cuelga de avance2/ para que la carpeta sea autocontenida.
RAIZ = Path(__file__).resolve().parent.parent
CORPUS_DIR = RAIZ / "corpus"
CHROMA_DIR = RAIZ / "chroma_db"
COLECCION = "calculo2"

# ---------------------------------------------------------------------------
# CHUNKING (justificacion con datos en docs/DECISIONES.md, seccion Chunking)
# ---------------------------------------------------------------------------
# Tamano objetivo de un chunk, en caracteres del texto ya limpio.
CHUNK_TAMANO = 1800
# Solapamiento: cola del chunk anterior que se repite al inicio del siguiente,
# para que una idea cortada en la frontera aparezca completa en uno de los dos.
CHUNK_SOLAPE = 200
# Un trozo menor que esto se fusiona con el siguiente de su documento: un
# fragmento de 60 caracteres casi no aporta contexto al modelo.
CHUNK_MINIMO = 200

# ---------------------------------------------------------------------------
# EMBEDDINGS (locales; nada sale de la maquina)
# ---------------------------------------------------------------------------
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"
# E5 se entreno con estos prefijos: sin ellos la calidad baja.
PREFIJO_CONSULTA = "query: "
PREFIJO_PASAJE = "passage: "
# max_seq_length del modelo (tokens). El texto a embeber no debe superarlo o
# el modelo lo trunca en silencio.
EMBEDDING_MAX_TOKENS = 512
# Que texto se convierte en vector (el texto GUARDADO siempre es el LaTeX
# completo): "crudo" | "limpio" | "verbal". Ver chunking.normalizar_para_embedding.
EMBEDDING_TEXTO_MODO = "limpio"

# ---------------------------------------------------------------------------
# RECUPERACION
# ---------------------------------------------------------------------------
TOP_K = 4
# Distancia coseno maxima (0 = identico, 2 = opuesto) para considerar que un
# fragmento "esta en el corpus". None = sin umbral. Se calibra con datos en
# la Fase 2 (distancias dentro vs fuera del corpus); no se fija a ojo.
UMBRAL_DISTANCIA: float | None = None

# ---------------------------------------------------------------------------
# GENERACION. Es lo unico que sale de la maquina: pregunta, historial y los
# top_k fragmentos recuperados.
#
# NO HAY PARAMETRO DE TEMPERATURA (ni top_p/top_k) en ninguna llamada: los
# modelos Claude actuales (Haiku 5.5, Sonnet 5.5) rechazan con 400 cualquier
# valor distinto del predeterminado, asi que se usa siempre el del modelo.
# La reproducibilidad en la evaluacion la da la cache en disco (llm.py).
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()   # anthropic | gemini | falso
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5").strip()          # tutor
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "claude-sonnet-5-5").strip()   # juez de Ragas
# max_tokens es OBLIGATORIO en cada llamada a Anthropic y se define de forma
# explicita. En los modelos 5.5 los tokens de razonamiento cuentan dentro de
# este tope, por eso la respuesta tiene margen amplio. Tambien protege la
# cuota cuando la app es publica.
MAX_TOKENS_RESPUESTA = 8000
MAX_TOKENS_CONDENSACION = 1024
# El juez descompone respuestas largas en afirmaciones y los modelos 5.5 razonan
# antes: con 4096 la corrida base fallo (LLMDidNotFinishException) en las 5
# respuestas de solucion completa. 16000 es el tope razonable sin streaming.
MAX_TOKENS_JUEZ = 16000
# Reintentos ante errores transitorios: maximo 3, SOLO ante 429 y 5xx.
REINTENTOS_API = 3
# Cache en disco de respuestas del LLM (clave = hash de modelo + prompt).
CACHE_LLM_DIR = RAIZ / ".cache_llm"
# Turnos reales del historial que se reenvian al modelo (pregunta y respuesta
# cuentan como 2). Mas historial = mas tokens en cada llamada.
# ---------------------------------------------------------------------------
# APP WEB (Streamlit). Limites que protegen la cuota de la API cuando la app es publica.
# ---------------------------------------------------------------------------
# top_k de la app. La Fase 2 no pudo demostrar que 8 sea mejor que 4 (dentro del
# ruido): 8 recupera la seccion correcta en 17/18 preguntas (4: 14/18) a cambio
# de ~17 % mas tokens de entrada. Decision abierta, reversible con esta linea.
APP_TOP_K = 8
# Preguntas por conversacion (cada una puede costar 2 llamadas: condensar + responder).
APP_MAX_PREGUNTAS = 15
# Longitud maxima de una pregunta (las del dataset llegan a ~600 caracteres).
APP_MAX_CARACTERES = 1500
# Caracteres del fragmento que se muestran en "Fuentes" (el LLM recibe el fragmento entero).
APP_FUENTE_VISTA = 700

HISTORIAL_MAX_TURNOS = 6
# Turnos del historial que se usan para condensar, y recorte de cada uno:
# basta el contexto inmediato; no hace falta reenviar respuestas enteras.
CONDENSACION_TURNOS = 4
CONDENSACION_RECORTE = 600
