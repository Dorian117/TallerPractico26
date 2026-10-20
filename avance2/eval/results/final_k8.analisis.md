# Análisis de la corrida `final_k8`

Tutor: `claude-haiku-5-5` · juez: `claude-sonnet-5-5` (juez final: True) · 23 preguntas · parámetros: `{"chunk_tamano": 1800, "chunk_solape": 200, "top_k": 8, "umbral": null, "embeddings": "intfloat/multilingual-e5-small", "texto_embedding": "limpio", "max_tokens_respuesta": 8000}`

## Métricas de Ragas (preguntas dentro del corpus)

| Métrica | Media | IC 95 % (bootstrap) | Desv. est. | Mínimo | n | sin puntaje |
|---|---|---|---|---|---|---|
| Faithfulness | 0.605 | [0.469, 0.731] | 0.287 | 0.000 | 18 | 0 |
| Answer relevancy | 0.943 | [0.931, 0.955] | 0.026 | 0.885 | 18 | 0 |
| Context precision | 0.697 | [0.522, 0.855] | 0.359 | 0.000 | 18 | 0 |
| Context recall | 0.731 | [0.556, 0.889] | 0.371 | 0.000 | 18 | 0 |

Métrica más baja: **Faithfulness** (0.605). Con ~18 preguntas y un LLM como juez los intervalos son anchos: dos métricas (o dos corridas) solo difieren con seguridad si sus IC no se solapan.

### Por tipo de pregunta

| Tipo | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| definicion | 6 | 0.75 | 0.95 | 0.88 | 0.92 |
| procedimiento | 5 | 0.54 | 0.93 | 0.44 | 0.70 |
| ejercicio | 4 | 0.58 | 0.95 | 0.71 | 0.71 |
| seguimiento | 3 | 0.44 | 0.93 | 0.75 | 0.44 |

### Evidencia: las preguntas con menor `faithfulness`

- **faithfulness = 0.000** · tipo `procedimiento` · «Intenté la integral de dx/(1 + sen x) multiplicando por el conjugado (1 − sen x), pero no sé cómo seguir despu»
  - archivo esperado en el top-8: **sí** (recuperó: c_digo_latex_de_sustit…ión con despeje algebraico): d=0.1018; c_digo_latex_de_sustit…ón con partición y despeje): d=0.1036; c_digo_latex_de_sustit…Identificación de $u$ y $du$ d=0.1092; c_digo_latex_de_sustit…ecta > Análisis de selección d=0.1107; c_digo_latex_de_sustit… de las Integrales > Caso h: d=0.1153; c_digo_latex_de_sustit…ales por Sustitución Directa d=0.1164; c_digo_latex_de_simpli… a los Ejercicios Propuestos d=0.117; c_digo_latex_de_sustit…de los Ejercicios Propuestos d=0.1172)
  - otras métricas: faithfulness=0.00, answer_relevancy=0.91, context_precision=0.00, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «Tu idea de multiplicar por el conjugado $(1-\sin x)$ es correcta y es el paso que necesitabas. Lo que falta es usar la identidad pitagórica para simplificar el denominador [2].  **1. Identificar la forma de la integral** El integrando es $\dfrac{1}{1+\sin x}$,…»
- **faithfulness = 0.034** · tipo `ejercicio` · «Intenté la integral de x² cos(x) por partes con u = x², pero queda una integral que también necesita partes y »
  - archivo esperado en el top-8: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.1052; c_digo_latex_de_sustit…ón con partición y despeje): d=0.1065; c_digo_latex_de_integr…Ejercicios de Identificación d=0.109; c_digo_latex_de_integr…imple (Contraste con Partes) d=0.1123; c_digo_latex_de_integr…de los Ejercicios Propuestos d=0.1124; c_digo_latex_de_integr…tegral del logaritmo natural d=0.1125; c_digo_latex_de_integr…style \int x^{3}\sin(x)\,dx$ d=0.114; c_digo_latex_de_sustit…de los Ejercicios Propuestos d=0.1141)
  - otras métricas: faithfulness=0.03, answer_relevancy=0.94, context_precision=0.20, context_recall=0.33
  - citas inválidas: [] · respuesta (inicio): «Tu elección $u = x^2$, $dv = \cos x\,dx$ es correcta, y también lo es $v = \sin x$. El problema está en el segundo paso, así que lo continúo desde tu planteamiento.  **1. Identificar la forma de la integral** Es un producto de una algebraica ($x^2$) por una tr…»
- **faithfulness = 0.350** · tipo `seguimiento` · «¿Y cómo decido cuál de las dos funciones va a ser u?»
  - archivo esperado en el top-8: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.106; c_digo_latex_de_integr…Ejercicios de Identificación d=0.1205; c_digo_latex_de_integr… a la Integración por Partes d=0.1211; c_digo_latex_de_integr…ca ILATE > Algoritmo General d=0.1241; c_digo_latex_de_integr…es > Deducción de la Fórmula d=0.1261; c_digo_latex_de_integr…tegral del logaritmo natural d=0.1261; c_digo_latex_de_sustit…ales por Sustitución Directa d=0.1263; c_digo_latex_de_sustit…alizar un Cambio de Variable d=0.1276)
  - otras métricas: faithfulness=0.35, answer_relevancy=0.88, context_precision=0.25, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «El criterio práctico es la prioridad **ILATE**: al elegir $u$, revisa primero si la función es inversa trigonométrica, luego logarítmica, algebraica, trigonométrica y, por último, exponencial. La función que aparezca antes en esa lista será $u$, y el resto ser…»
- **faithfulness = 0.379** · tipo `procedimiento` · «Intenté resolver la integral de 1/(x² · raíz de (9 − x²)) con la sustitución x = 3 sen θ y llegué a la integra»
  - archivo esperado en el top-8: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.1048; c_digo_latex_de_simpli… a los Ejercicios Propuestos d=0.1086; c_digo_latex_de_sustit…ión con despeje algebraico): d=0.1098; c_digo_latex_de_sustic…de los Ejercicios Propuestos d=0.1106; c_digo_latex_de_sustic…o 3: Caso $\sqrt{x^2 - a^2}$ d=0.1109; c_digo_latex_de_sustit…ales por Sustitución Directa d=0.1111; c_digo_latex_de_sustit…ón con partición y despeje): d=0.1118; c_digo_latex_de_sustic…o 1: Caso $\sqrt{a^2 - x^2}$ d=0.1134)
  - otras métricas: faithfulness=0.38, answer_relevancy=0.93, context_precision=0.12, context_recall=0.50
  - citas inválidas: [] · respuesta (inicio): «Tu planteamiento es correcto: $x = 3\sin\theta$ es la sustitución adecuada para $\sqrt{9-x^2}$ [1], y tu integral en $\theta$ está bien armada. Te muestro cómo termina, partiendo de ese punto.  **1. Identificar la forma de la integral** $\int \frac{dx}{x^2\sqr…»

## Fuera de corpus (métricas propias, aparte de Ragas)

- n = 5 · **tasa de abstención correcta = 1.0** · **tasa de alucinación = 0.0**
- Abstención indebida (dentro del corpus, respondió «no encontré»): 0.0
- Respuestas con citas a fragmentos inexistentes: 0

## Calibración del umbral de distancia (2.3)

- Top-1 **dentro** del corpus (n=18): min 0.0887 · mediana 0.1048 · max 0.1422
- Top-1 **fuera** del corpus (n=5): min 0.1313 · mediana 0.1511 · max 0.1820
- AUC (prob. de que una pregunta del corpus tenga mejor top-1 que una de fuera): **0.97**
- Margen (min fuera − max dentro): **-0.0109** (SOLAPADAS: ningun umbral separa sin errores)
- Mejor umbral por exactitud balanceada: 0.1295 (exactitud 0.94)

| umbral | dentro retenidas | fuera rechazadas | exactitud balanceada |
|---|---|---|---|
| 0.0887 | 6% | 100% | 0.53 |
| 0.0907 | 11% | 100% | 0.56 |
| 0.0966 | 17% | 100% | 0.58 |
| 0.1016 | 22% | 100% | 0.61 |
| 0.1017 | 28% | 100% | 0.64 |
| 0.1018 | 33% | 100% | 0.67 |
| 0.1022 | 39% | 100% | 0.69 |
| 0.1023 | 44% | 100% | 0.72 |
| 0.1048 | 50% | 100% | 0.75 |
| 0.1049 | 56% | 100% | 0.78 |
| 0.1050 | 61% | 100% | 0.81 |
| 0.1052 | 67% | 100% | 0.83 |
| 0.1054 | 72% | 100% | 0.86 |
| 0.1060 | 78% | 100% | 0.89 |
| 0.1216 | 83% | 100% | 0.92 |
| 0.1295 | 89% | 100% | 0.94 |
| 0.1313 | 89% | 80% | 0.84 |
| 0.1327 | 94% | 80% | 0.87 |
| 0.1405 | 94% | 60% | 0.77 |
| 0.1422 | 100% | 60% | 0.80 |
| 0.1511 | 100% | 40% | 0.70 |
| 0.1669 | 100% | 20% | 0.60 |
| 0.1820 | 100% | 0% | 0.50 |
