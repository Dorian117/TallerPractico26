# Guía de despliegue del Tutor de Cálculo II

Todo lo de esta guía lo haces **tú** en tus cuentas. No tienes que pasarme claves ni contraseñas
(ni a mí ni a nadie por chat), y yo no hago `commit` ni `push`.

## 1. Dónde publicarlo (comprobado en la documentación vigente el 2026-10-16)

| | Streamlit Community Cloud | Hugging Face Spaces |
|---|---|---|
| ¿Acepta una app Streamlit? | **Sí** (es su plataforma) | **No directamente**: el selector de SDK solo ofrece `gradio`, `docker` y `static` |
| Coste para ti | Gratis | Los Spaces *Gradio* y *Docker* "requieren un plan de pago para crearse" (PRO, 9 USD/mes, según su página de precios). Los *static* son gratis, pero no ejecutan Python |
| Recursos | ≈ 2 CPU, ≈ 2,7 GB de RAM, 50 GB de disco (cifras de **febrero de 2024**, "pueden cambiar sin aviso") | CPU Basic: 2 vCPU, 16 GB RAM, 50 GB de disco (no persistente) |
| Inactividad | Se duerme tras 12 h sin tráfico; cualquiera con acceso lo despierta con un clic | Se duerme tras un tiempo sin uso en hardware gratuito (la documentación no da la cifra) |
| Secretos | Panel *Secrets* (formato TOML) | *Settings → Secrets*, expuestos como variables de entorno |
| Versión de Python | Se elige en *Advanced settings*; por defecto 3.12 | `python_version` en el README del Space (solo para SDK gradio) / lo que pongas en el Dockerfile |

Fuentes: documentación de Streamlit (*deploy*, *app-dependencies*, *manage-your-app*,
*secrets-management*) y de Hugging Face (*spaces-overview*, *spaces-config-reference*,
*pricing*).

**Dificultad encontrada (hay que decidirla):** mi plan inicial era un Space de Hugging Face
con SDK `streamlit`. Esa opción **ya no existe** en la documentación actual, y los Spaces
con Docker (la vía para correr Streamlit allí) piden un plan de pago. Por eso esta guía
usa **Streamlit Community Cloud** como ruta principal. Si tienes PRO o prefieres Hugging
Face igualmente, dímelo y preparo un `Dockerfile` (no lo escribí: no puedo probarlo aquí y
no quiero dejarte un archivo sin verificar).

## 2. Qué se publicaría (si el repositorio es público)

Comprueba esta lista antes de hacer `push`. **Todo lo que no esté ignorado por git se
publica**, incluido el contenido de `corpus/`:

- `corpus/` (6 archivos, ≈ 66 KB de apuntes en LaTeX):
  - `c_digo_latex_de_antiderivaci_n.txt`
  - `c_digo_latex_de_simplificaci_n_algebraica.txt`
  - `c_digo_latex_de_sustitucion.txt`
  - `c_digo_latex_de_integraci_n_por_partes.txt`
  - `c_digo_latex_de_Areas_entre-Curvas.txt` (su contenido real son integrales trigonométricas, y está truncado)
  - `c_digo_latex_de_susticion_trigonometrica.txt`
- Código (`app.py`, `rag/`, `eval/`, `tests/`), `docs/`, y los resultados de evaluación en
  `eval/results/` (incluyen las preguntas y respuestas del dataset).
- Además, el prompt del tutor incluye **ejemplos copiados del corpus** (`rag/prompts.py`).

**No se publica** (git lo ignora; compruébalo con el comando de abajo): `.env`,
`.streamlit/secrets.toml`, `.venv/`, `chroma_db/` (el índice se reconstruye solo en el
primer arranque), `.cache_llm/`.

Si los apuntes tienen derechos de autor de terceros o no quieres hacerlos públicos,
**no uses un repositorio público**: pregunta antes por la opción de repositorio privado
(no está verificada aquí que Community Cloud la permita en la cuenta gratuita).

```bash
git status --short --ignored avance2
git check-ignore -v avance2/.env avance2/.streamlit/secrets.toml avance2/chroma_db avance2/.cache_llm
git ls-files avance2 | grep -E "\.env$|secrets\.toml$" || echo "ningún archivo secreto con seguimiento"
```

## 3. Pasos (Streamlit Community Cloud)

1. **Límite de gasto primero.** En la consola de Anthropic, fija un límite de gasto mensual
   para la clave que usará la app (comprueba que tu plan lo ofrece). Una app pública
   gasta *tu* presupuesto cada vez que alguien pregunta. La app ya limita 15 preguntas por
   conversación y `max_tokens`, pero **no puede impedir que alguien abra muchas
   conversaciones**: el límite de gasto de la consola es la protección real. Si se agota,
   la app muestra "El servicio alcanzó su límite de uso".
2. **Sube el código a GitHub** (esto lo haces tú). Con la carpeta `avance2/` dentro de un
   repositorio.
3. Entra en <https://share.streamlit.io> con tu cuenta de GitHub → **Create app** → *Yup, I have an app*.
4. Rellena: repositorio, rama (`main`) y **archivo principal: `avance2/app.py`**.
   *(La documentación pide "file path" pero no confirma de forma explícita que pueda estar
   en una subcarpeta; si no lo acepta, mueve el contenido de `avance2/` a la raíz del repo.)*
5. **Advanced settings**: Python **3.12** (el valor por defecto) y, en **Secrets**, pega:

   ```toml
   ANTHROPIC_API_KEY = "tu-clave-aquí"
   ```

   Escribe la clave solo en ese cuadro de la web. La app la lee con `os.getenv` o
   `st.secrets` (`rag/llm.py::_secreto`).
6. **Deploy.** El primer arranque tarda: instala dependencias, descarga el modelo de
   embeddings (≈ 470 MB desde Hugging Face) y construye el índice (88 chunks).
7. Cuando cargue, haz la prueba de la sección 5.

**Dependencias:** Community Cloud busca el archivo primero en la carpeta del archivo
principal (`avance2/requirements.txt`) y luego en la raíz; usa **solo el primero que
encuentre**. Ese archivo ya está.

**Riesgos que NO pude verificar sin desplegar:**
- En Linux, `pip install torch` desde PyPI trae la versión con CUDA (mucho más grande).
  Si la instalación es lenta o falla, la alternativa habitual es usar el índice de torch
  para CPU; no está comprobado que Community Cloud acepte esa opción dentro del
  `requirements.txt`, así que no la añadí.
- Memoria: en mi prueba local, con el modelo cargado y 4 turnos, el proceso usó ≈ 1,34 GB
  (working set). Cabe en los ≈ 2,7 GB (cifra de 2024), pero con margen justo si hay
  varias sesiones a la vez (el modelo y el índice se comparten entre sesiones).
- Arranque en frío: en mi equipo la primera carga tardó ≈ 70 s.

## 4. Variables opcionales

| Variable (Secrets o entorno) | Efecto |
|---|---|
| `ANTHROPIC_API_KEY` | **Obligatoria** |
| `LLM_MODEL` | Modelo del tutor (por defecto `claude-haiku-5-5`) |

## 5. Prueba local de 4 turnos (hecha) y capturas a entregar

Ejecución local real con Haiku 5.5 (`streamlit run app.py`):

| Turno | Pregunta | Resultado |
|---|---|---|
| 1 | Ejercicio con intento previo y petición de solución completa | Resolvió y verificó derivando; resultado correcto; 13 425 tokens de entrada, 2 114 de salida, 15,2 s |
| 2 | Seguimiento: «¿Y si en lugar de $4x^2$ fuera $9x^2$?» | Entendió a qué integral se refería (condensación) y respondió con pistas; 14 881 / 1 434, 11,2 s |
| 3 | Fuera del corpus (volumen por discos) | «No encontré esto en el material del curso…»; no inventó; 15 312 / 491, 3,8 s |
| 4 | Definición (ILATE) | Respuesta con citas `[n]` y ejemplo del material; 14 968 / 859, 4,7 s |

Total de la conversación: 58 586 tokens de entrada y 4 898 de salida (sin contar la llamada
de condensación del turno 2). Coste estimado con la misma tarifa de la §9 de `DECISIONES.md`:
del orden de 0,01 USD; revísalo en tu consola.

**Capturas que conviene entregar** (hazlas tú en la app desplegada):
1. Pantalla inicial con el saludo y la barra lateral.
2. Turno 1 con la fórmula renderizada y el panel **Fuentes** abierto (distancias visibles).
3. Turno 2 (seguimiento) mostrando que mantiene el contexto.
4. Turno 3 (abstención fuera del corpus).
5. Barra lateral con contador de preguntas y tokens, y después de **Nueva conversación**.

## 6. Comprobaciones de seguridad hechas

- Búsqueda del **valor literal** de las claves de los `.env` (sin imprimirlas) y de los
  formatos `sk-ant-`, `AIza`, `AQ.`, `hf_`, `ghp_`: **no aparecen en ningún archivo con
  seguimiento ni sin él (salvo los `.env`), ni en los 4 commits del historial**. La única
  coincidencia de patrón es una clave **falsa** en `tests/test_eval.py`.
- `.env`, `.streamlit/secrets.toml`, `chroma_db/` y `.cache_llm/` están ignorados por git.
- Telemetría de Streamlit desactivada (`.streamlit/config.toml`: `gatherUsageStats = false`).

Repite la búsqueda justo antes de hacer `push` si cambias algo.
