## 1.2 Extracción estructurada

### Esquema

| Campo | Tipo | Descripción |
|---|---|---|
| `enunciado` | texto | Integral planteada al inicio, en texto o LaTeX. |
| `tipo_integral` | enum: definida \| indefinida | Definida si tiene límites de integración. |
| `tecnica` | enum: sustitucion \| partes \| sustitucion_trigonometrica \| fracciones_parciales \| directa \| otra | Técnica principal que usa el desarrollo. |
| `resultado_final` | texto | Resultado escrito al final del último paso. |
| `numero_pasos` | entero (0–30) | Cantidad de pasos numerados ("Paso n") escritos en la imagen. |

### Tabla de aciertos

| Imagen | Campo | Valor real | Valor extraído | ¿Correcto? |
|---|---|---|---|---|
| ej1_sust_trig.png | enunciado | \int \frac{dx}{\sqrt{4x^2+1}} | \int \frac{dx}{\sqrt{4x^2 + 1}} | Sí |
| ej1_sust_trig.png | tipo_integral | indefinida | indefinida | Sí |
| ej1_sust_trig.png | tecnica | sustitucion_trigonometrica | sustitucion_trigonometrica | Sí |
| ej1_sust_trig.png | resultado_final | \frac{1}{2}\ln\|\sqrt{4x^2+1}+2x\|+C | \frac{1}{2}\ln\|\sqrt{4x^2 + 1} + 2x\| + C | Sí |
| ej1_sust_trig.png | numero_pasos | 4 | 4 | Sí |
| ej2_partes.png | enunciado | \int x e^{x} dx | ∫ x e^x dx | Sí |
| ej2_partes.png | tipo_integral | indefinida | indefinida | Sí |
| ej2_partes.png | tecnica | partes | partes | Sí |
| ej2_partes.png | resultado_final | x e^{x} - e^{x} + C | x e^x - e^x + C | Sí |
| ej2_partes.png | numero_pasos | 4 | 4 | Sí |
| ej3_definida_sust.png | enunciado | \int_{0}^{1} 2x\cos(x^{2}) dx | ∫₀¹ 2x cos(x²) dx | No |
| ej3_definida_sust.png | tipo_integral | definida | definida | Sí |
| ej3_definida_sust.png | tecnica | sustitucion | sustitucion | Sí |
| ej3_definida_sust.png | resultado_final | \sin(1) | sin(1) | Sí |
| ej3_definida_sust.png | numero_pasos | 3 | 3 | Sí |

**Porcentaje de aciertos:** 14 de 15 campos = 93.3 %

> Nota: `enunciado` y `resultado_final` se comparan tras normalizar la notación (LaTeX/Unicode). Revisa a mano los "No" de esos dos campos: pueden ser equivalentes.
