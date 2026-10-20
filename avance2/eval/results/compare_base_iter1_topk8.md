# Comparación `base` → `iter1_topk8`


**Parámetros que cambiaron:** `{"top_k": [4, 8]}`

| Métrica | Antes | Después | Δ | ¿Dentro del ruido? | Veredicto |
|---|---|---|---|---|---|
| Faithfulness | 0.540 | 0.586 | +0.046 | sí | sin cambio claro |
| Answer relevancy | 0.825 | 0.935 | +0.110 | sí | sin cambio claro |
| Context precision | 0.718 | 0.680 | -0.037 | sí | bajó (dentro del ruido) |
| Context recall | 0.667 | 0.722 | +0.056 | sí | sin cambio claro |

## Fuera de corpus

- Abstención correcta: 1.0 → 1.0
- Alucinación: 0.0 → 0.0
- Abstención indebida (dentro del corpus): 0.05555555555555555 → 0.0
- Respuestas con citas inválidas: 0 → 0

## Diferencias pareadas por pregunta (n = 18)

| Métrica | preguntas que mejoran | empeoran | iguales | Δ medio pareado |
|---|---|---|---|---|
| Faithfulness | 10 | 6 | 2 | +0.046 |
| Answer relevancy | 9 | 2 | 7 | +0.110 |
| Context precision | 1 | 5 | 12 | -0.037 |
| Context recall | 3 | 1 | 14 | +0.056 |