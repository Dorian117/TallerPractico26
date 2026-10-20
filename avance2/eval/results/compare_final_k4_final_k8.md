# Comparación `final_k4` → `final_k8`


**Parámetros que cambiaron:** `{"top_k": [4, 8]}`

| Métrica | Antes | Después | Δ | ¿Dentro del ruido? | Veredicto |
|---|---|---|---|---|---|
| Faithfulness | 0.546 | 0.605 | +0.059 | sí | sin cambio claro |
| Answer relevancy | 0.890 | 0.943 | +0.053 | sí | sin cambio claro |
| Context precision | 0.718 | 0.697 | -0.020 | sí | bajó (dentro del ruido) |
| Context recall | 0.685 | 0.731 | +0.046 | sí | sin cambio claro |

## Fuera de corpus

- Abstención correcta: 1.0 → 1.0
- Alucinación: 0.0 → 0.0
- Abstención indebida (dentro del corpus): 0.05555555555555555 → 0.0
- Respuestas con citas inválidas: 0 → 0

## Diferencias pareadas por pregunta (n = 18)

| Métrica | preguntas que mejoran | empeoran | iguales | Δ medio pareado |
|---|---|---|---|---|
| Faithfulness | 10 | 8 | 0 | +0.059 |
| Answer relevancy | 4 | 3 | 11 | +0.053 |
| Context precision | 2 | 3 | 13 | -0.020 |
| Context recall | 2 | 0 | 16 | +0.046 |