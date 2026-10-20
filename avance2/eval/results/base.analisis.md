# Análisis de la corrida `base`

Tutor: `claude-haiku-5-5` · juez: `claude-haiku-5-5` (juez final: False) · 23 preguntas · parámetros: `{"chunk_tamano": 1800, "chunk_solape": 200, "top_k": 4, "umbral": null, "embeddings": "intfloat/multilingual-e5-small", "texto_embedding": "limpio", "max_tokens_respuesta": 8000}`

## Métricas de Ragas (preguntas dentro del corpus)

| Métrica | Media | IC 95 % (bootstrap) | Desv. est. | Mínimo | n | sin puntaje |
|---|---|---|---|---|---|---|
| Faithfulness | 0.540 | [0.409, 0.671] | 0.287 | 0.000 | 18 | 0 |
| Answer relevancy | 0.825 | [0.669, 0.934] | 0.302 | 0.000 | 18 | 0 |
| Context precision | 0.718 | [0.519, 0.894] | 0.406 | 0.000 | 18 | 0 |
| Context recall | 0.667 | [0.444, 0.870] | 0.457 | 0.000 | 18 | 0 |

Métrica más baja: **Faithfulness** (0.540). Con ~18 preguntas y un LLM como juez los intervalos son anchos: dos métricas (o dos corridas) solo difieren con seguridad si sus IC no se solapan.

### Por tipo de pregunta

| Tipo | n | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|
| definicion | 6 | 0.78 | 0.95 | 0.93 | 0.94 |
| procedimiento | 5 | 0.45 | 0.73 | 0.42 | 0.60 |
| ejercicio | 4 | 0.37 | 0.69 | 0.75 | 0.50 |
| seguimiento | 3 | 0.44 | 0.91 | 0.75 | 0.44 |

### Evidencia: las preguntas con menor `faithfulness`

- **faithfulness = 0.000** · tipo `procedimiento` · «Intenté resolver la integral de 1/(x² · raíz de (9 − x²)) con la sustitución x = 3 sen θ y llegué a la integra»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.1048; c_digo_latex_de_simpli… a los Ejercicios Propuestos d=0.1086; c_digo_latex_de_sustit…ión con despeje algebraico): d=0.1098; c_digo_latex_de_sustic…de los Ejercicios Propuestos d=0.1106)
  - otras métricas: faithfulness=0.00, answer_relevancy=0.91, context_precision=0.00, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «Tu planteamiento es correcto. Con $x = 3\sin\theta$ obtuviste  $$\int \frac{3\cos\theta}{9\sin^2\theta\,\sqrt{9-9\sin^2\theta}}\,d\theta,$$  que es el resultado de sustituir $dx = 3\cos\theta\,d\theta$ y $x^2 = 9\sin^2\theta$ en la integral original. Lo que te…»
- **faithfulness = 0.065** · tipo `ejercicio` · «Intenté la integral de x² cos(x) por partes con u = x², pero queda una integral que también necesita partes y »
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.1052; c_digo_latex_de_sustit…ón con partición y despeje): d=0.1065; c_digo_latex_de_integr…Ejercicios de Identificación d=0.109; c_digo_latex_de_integr…imple (Contraste con Partes) d=0.1123)
  - otras métricas: faithfulness=0.07, answer_relevancy=0.92, context_precision=0.00, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «Tu elección $u = x^2$, $dv = \cos(x)\,dx$ es correcta: la parte algebraica es $u$ y la trigonométrica es $dv$, igual que en los ejemplos de identificación del material, como $(x+2)^2\sin(x)$ con $u=(x+2)^2$ [3]. Lo que te falta es completar el segundo paso por…»
- **faithfulness = 0.231** · tipo `ejercicio` · «Intenté hallar la antiderivada general de f(x) = 3 sen x + (2x⁵ − 2√x)/x + 11x⁻³, pero no sé cómo tratar el co»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_antide…a de Antiderivadas Generales d=0.0966; c_digo_latex_de_antide…Actividad Inicial d=0.1002; c_digo_latex_de_antide…Actividad Inicial d=0.1052; c_digo_latex_de_antide…ada) > Solución paso a paso: d=0.1081)
  - otras métricas: faithfulness=0.23, answer_relevancy=0.00, context_precision=1.00, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «Todavía no veo tu trabajo escrito, así que te doy pistas para que avances tú. Empieza por el cociente.  1. **Simplifica antes de integrar.** Todo el paréntesis $(2x^5 - 2\sqrt{x})$ está dividido entre $x$. ¿Qué obtienes si divides cada término del numerador en…»
- **faithfulness = 0.250** · tipo `seguimiento` · «¿Y cómo decido cuál de las dos funciones va a ser u?»
  - archivo esperado en el top-4: **sí** (recuperó: c_digo_latex_de_sustit…ecta > Análisis de selección d=0.106; c_digo_latex_de_integr…Ejercicios de Identificación d=0.1205; c_digo_latex_de_integr… a la Integración por Partes d=0.1211; c_digo_latex_de_integr…ca ILATE > Algoritmo General d=0.1241)
  - otras métricas: faithfulness=0.25, answer_relevancy=0.88, context_precision=0.25, context_recall=0.00
  - citas inválidas: [] · respuesta (inicio): «La respuesta está en la regla ILATE: se da prioridad descendente para elegir $u$ entre las funciones del integrando, en este orden: Inversas trigonométricas, Logarítmicas, Algebraicas, Trigonométricas y Exponenciales [2][4]. La función que aparezca primero en …»

## Fuera de corpus (métricas propias, aparte de Ragas)

- n = 5 · **tasa de abstención correcta = 1.0** · **tasa de alucinación = 0.0**
- Abstención indebida (dentro del corpus, respondió «no encontré»): 0.05555555555555555
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
