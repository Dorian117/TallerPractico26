"""Genera las imágenes de ejemplo del dominio (Cálculo II) en imagenes/.

Son hojas de ejercicios con fórmulas renderizadas, con una ligera rotación y
ruido para parecerse a una foto. Sirven para que el taller funcione de principio a
fin. LO IDEAL es sustituirlas o complementarlas con fotos reales de apuntes
manuscritos (ver README). Si cambias las imágenes, actualiza
imagenes/verdad_terreno.json con los valores que TÚ lees en ellas.

Requiere: pip install matplotlib pillow numpy
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "imagenes"
SALIDA.mkdir(exist_ok=True)
random.seed(7)

PAPEL = "#fbf7e8"
TINTA = "#1d2a5b"


def hoja(nombre: str, titulo: str, lineas: list[str], ancho=7.6, alto=None, angulo=0.0):
    alto = alto or (1.2 + 0.78 * len(lineas))
    fig = plt.figure(figsize=(ancho, alto), dpi=130, facecolor=PAPEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # renglones de cuaderno
    n = int(alto * 4)
    for i in range(1, n):
        ax.axhline(i / n, color="#cdd8ee", lw=0.6, zorder=0)
    ax.text(0.04, 1 - 0.55 / alto, titulo, fontsize=15, color=TINTA, va="center", weight="bold")
    y = 1 - 1.25 / alto
    for ln in lineas:
        ax.text(0.06, y, ln, fontsize=15, color=TINTA, va="center")
        y -= 0.78 / alto
    ruta = SALIDA / nombre
    fig.savefig(ruta, facecolor=PAPEL)
    plt.close(fig)
    # efecto "foto": rotación leve + ligero desenfoque
    im = Image.open(ruta).convert("RGB")
    if angulo:
        im = im.rotate(angulo, expand=True, fillcolor=(236, 232, 220), resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(0.5))
    im.save(ruta)
    print("creada", ruta.name)


def tabla(nombre: str):
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=130, facecolor=PAPEL)
    ax.axis("off")
    ax.set_title("Casos estándar de sustitución trigonométrica", color=TINTA,
                 fontsize=14, weight="bold", loc="left")
    filas = [
        [r"$\sqrt{a^2-x^2}$", r"$x = a\,\sin\theta$", r"$\sqrt{a^2-x^2}=a\cos\theta$"],
        [r"$\sqrt{a^2+x^2}$", r"$x = a\,\tan\theta$", r"$\sqrt{a^2+x^2}=a\sec\theta$"],
        [r"$\sqrt{x^2-a^2}$", r"$x = a\,\sec\theta$", r"$\sqrt{x^2-a^2}=a\tan\theta$"],
    ]
    t = ax.table(cellText=filas, colLabels=["Expresión", "Sustitución", "Resultado"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(14)
    t.scale(1, 2.3)
    for (r, c), cel in t.get_celld().items():
        cel.set_facecolor("#eef2fb" if r == 0 else PAPEL)
        cel.set_edgecolor("#9aa8d4")
    fig.savefig(SALIDA / nombre, facecolor=PAPEL)
    plt.close(fig)
    im = Image.open(SALIDA / nombre).convert("RGB").filter(ImageFilter.GaussianBlur(0.5))
    im.save(SALIDA / nombre)
    print("creada", nombre)


def main():
    hoja("ej1_sust_trig.png", "Ejercicio 1",
         [r"$\int \dfrac{dx}{\sqrt{4x^2+1}}$",
          r"Paso 1:  $2x=\tan\theta,\quad dx=\frac{1}{2}\sec^2\theta\,d\theta$",
          r"Paso 2:  $\sqrt{4x^2+1}=\sec\theta$",
          r"Paso 3:  $=\frac{1}{2}\int \sec\theta\,d\theta$",
          r"Paso 4:  $=\frac{1}{2}\ln|\sec\theta+\tan\theta|+C=\frac{1}{2}\ln|\sqrt{4x^2+1}+2x|+C$"],
         ancho=9.0, angulo=1.2)

    hoja("ej2_partes.png", "Ejercicio 2",
         [r"$\int x\,e^{x}\,dx$",
          r"Paso 1:  $u=x,\quad dv=e^{x}dx$",
          r"Paso 2:  $du=dx,\quad v=e^{x}$",
          r"Paso 3:  $\int x\,e^{x}dx = x\,e^{x}-\int e^{x}dx$",
          r"Paso 4:  $= x\,e^{x}-e^{x}+C$"],
         angulo=-1.0)

    hoja("ej3_definida_sust.png", "Ejercicio 3",
         [r"$\int_{0}^{1} 2x\cos(x^{2})\,dx$",
          r"Paso 1:  $u=x^{2},\quad du=2x\,dx$",
          r"Paso 2:  si $x=0\Rightarrow u=0$;  si $x=1\Rightarrow u=1$",
          r"Paso 3:  $=\int_{0}^{1}\cos u\,du=\sin u\,|_{0}^{1}=\sin(1)$"],
         angulo=0.8)

    tabla("ej4_tabla_casos.png")

    # Mismo ejercicio 2 resuelto con un ERROR DE SIGNO en el paso 3 -> 4 (para 1.3)
    hoja("ej5_error_signo.png", "Ejercicio 2 (resuelto por otro estudiante)",
         [r"$\int x\,e^{x}\,dx$",
          r"Paso 1:  $u=x,\quad dv=e^{x}dx$",
          r"Paso 2:  $du=dx,\quad v=e^{x}$",
          r"Paso 3:  $\int x\,e^{x}dx = x\,e^{x}-\int e^{x}dx$",
          r"Paso 4:  $= x\,e^{x}+e^{x}+C$"],
         angulo=1.6)


if __name__ == "__main__":
    main()
