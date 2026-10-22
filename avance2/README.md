# Avance 2 — Tutor de Cálculo II con RAG

Pipeline RAG sobre apuntes de Cálculo II (técnicas de integración), con
embeddings locales, índice Chroma propio y generación con Claude (Anthropic). Decisiones y
cifras: [`docs/DECISIONES.md`](docs/DECISIONES.md).

**Estado:** Fase 1 (pipeline RAG) completa, con la demo conversacional ejecutada
(LLM por defecto: Anthropic, Haiku 5.5). **Fase 2 completa**: dataset aprobado (23
preguntas), evaluación con Ragas (juez Sonnet 5.5 en las corridas finales), calibración del
umbral, diagnóstico e iteración de un parámetro (`top_k` 4 → 8); resultados en
`docs/DECISIONES.md` §8–9 y `eval/results/`. **Fase 3** (chat web con Streamlit): app
construida y probada en local con 4 turnos reales; el despliegue lo haces tú siguiendo
[`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md) (decisión abierta: dónde desplegar).

## Privacidad

- Los embeddings se calculan **localmente** (`intfloat/multilingual-e5-small`) y el índice es una carpeta tuya (`chroma_db/`).
- Al proveedor del LLM (Anthropic por defecto) solo se envían: la pregunta, el historial, los `top_k` fragmentos recuperados y el texto fijo del prompt (que incluye ejemplos copiados del corpus). Nunca el corpus completo ni archivos.
- La clave solo va por variable de entorno / `.env` / secrets. Nunca en el repo.
- Telemetría de Chroma y Ragas desactivada. Detalle y puntos a comprobar en `docs/DECISIONES.md` (sección 11).

## Instalación (Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-eval.txt     # ejecución + evaluación (Python 3.12)
# o solo ejecución: pip install -r requirements.txt
copy .env.example .env                    # y completa ANTHROPIC_API_KEY
```

## Uso

| Comando | Qué hace |
|---|---|
| `python -m rag.ingest` | Reconstruye **todo** el índice desde `corpus/` (idempotente). `--muestras 3` muestra chunks; `--sin-indexar` solo informe |
| `python -m rag.retrieve "pregunta"` | Muestra los fragmentos recuperados y sus distancias (no llama al LLM) |
| `python -m rag.cli` | Tutor interactivo por consola (`nueva` reinicia, `salir` termina) |
| `streamlit run app.py` | Chat web (necesita `ANTHROPIC_API_KEY` en `.env` o en `.streamlit/secrets.toml`; el primer arranque tarda ≈ 1 min) |
| `python -m tests.test_app` | Tests de la app con LLM falso (0 gasto) |
| `python -m rag.cli --demo` | Demo guionizada de 3 turnos (**llamadas reales al LLM**, con caché en disco) |
| `python -m tests.test_llm` | Tests de la capa LLM sin LLM real (`--live` añade 1 llamada mínima real) |
| `python -m tests.test_eval` | Tests de la evaluación sin LLM real |
| `python -m eval.run_eval --nombre x` | Evaluación en modo OFFLINE (caché o LLM falso, 0 gasto; resultado no válido) |
| `python -m eval.run_eval --nombre x --live [--limite N] [--juez-final]` | Evaluación real; el juez es el modelo barato salvo `--juez-final` |
| `python -m eval.analisis <corrida>` | Tabla (Markdown + LaTeX), gráfico PNG, IC, evidencia y calibración del umbral (sin API) |
| `python -m eval.compare <antes> <despues>` | Tabla antes/después con deltas, ruido y diferencias pareadas (sin API) |
| `python -m eval.bench_retrieval base` | Qué cambio de recuperación rescata más preguntas (sin API) |
| `python -m tests.smoke_readers` | Pruebas locales de lectores y chunker (sin red) |
| `python -m eval.bench_embeddings` | Benchmark de modelos de embeddings (sin API) |

## Estructura

```
app.py             chat web (Streamlit)
.streamlit/        config.toml (sin telemetría); secrets.toml.example (copiar a secrets.toml, ignorado)
corpus/            apuntes (.txt con LaTeX; también admite pdf/docx/md/html/tex)
rag/config.py      TODOS los parámetros (chunking, modelo, top_k, umbral, LLM)
rag/readers.py     un lector por formato
rag/chunking.py    troceado con límite en tokens; fórmulas indivisibles
rag/embeddings.py  E5 local (prefijos query:/passage:)
rag/store.py       Chroma persistente, distancia coseno
rag/retrieve.py    top_k + distancias (umbral pendiente de calibrar, Fase 2)
rag/llm.py         capa de LLM: Anthropic (defecto) / Gemini / falso; reintentos, límite de uso, caché
rag/prompts.py     system prompt, few-shot (user/assistant), delimitadores, condensación
rag/generate.py    condensar → recuperar → generar
rag/cli.py         tutor por consola
eval/              run_eval.py, juez.py (juez Anthropic para Ragas), bench_embeddings.py
docs/DECISIONES.md decisiones, alternativas, cifras reales, privacidad
docs/DESPLIEGUE.md guía de despliegue, qué se publica, prueba de 4 turnos, capturas
docs/GUIA_CODIGO.md qué hace cada parte del código y por qué (para entenderlo a fondo)
docs/GLOSARIO.md   diccionario de términos (RAG, embedding, chunk, métricas, errores HTTP...)
```

## Avisos sobre el corpus

Se auditaron todas las respuestas derivando: 4 erróneas se **corrigieron**, 2 más (partes e, i)
quedan pendientes de tu decisión; `Areas_entre-Curvas.txt` está truncado y su nombre no
corresponde a su contenido (integrales trigonométricas). Ver `docs/DECISIONES.md`, sección 1.
