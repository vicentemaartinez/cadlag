---
name: càdlàg
description: Suizo con wabi-sabi. Papel con fibra, una sola familia sans con eje de ancho, filetes finos, un rojo apagado como único acento. La matriz de posiciones es el centro.
colors:
  paper: "#ECE7DD"        # fondo (claro)
  paper-2: "#E3DDD1"
  paper-3: "#D5CDBD"
  ink: "#1D1B18"
  ink-2: "#57514A"
  ink-3: "#8B8378"
  rule: "#CCC3B3"
  beni: "#8C2B2A"         # acento único: rojo apagado
  ai: "#2E4B5E"           # índigo: zonas altas, subidas, título
  kaki: "#9C7838"         # ocre: zonas intermedias
  heat-0: "#F1DED4"       # rampa de la matriz (OKLab, t = (p/0.6)^0.62)
  heat-1: "#7E2128"
  dark-paper: "#2A2622"   # oscuro: carbón cálido, nunca negro
  dark-paper-2: "#322D28"
  dark-paper-3: "#3E3832"
  dark-ink: "#EDE6DA"
  dark-ink-2: "#B8AE9F"
  dark-ink-3: "#8A8175"
  dark-rule: "#4A433B"
  dark-beni: "#E07D6E"
  dark-ai: "#93B6CB"
  dark-kaki: "#D3AF6C"
  dark-heat-0: "#3A312C"
  dark-heat-1: "#E27A63"
typography:
  sans: "Archivo (wdth 62–125, wght 100–900), Helvetica Neue, Arial, sans-serif"
  note: "Zen Old Mincho, Hiragino Mincho ProN, Yu Mincho, Georgia, serif"
  wordmark: { family: sans, weight: 700, wdth: 112–118, tracking: "-0.045em a -0.065em", case: lower }
  display-number: { family: sans, weight: 200, tracking: "-0.05em" }   # porcentajes grandes
  h2: { family: sans, weight: 600, wdth: 96, tracking: "-0.035em" }
  body: { family: sans, size: 15px, weight: 400, line-height: 1.5 }
  meta: { family: sans, size: 12.5px, weight: 500, color: ink-2 }      # sin mayúsculas espaciadas
  note: { family: note, size: 13px }                                   # solo notas al margen
rounded: { none: 0, card: 3px, pill: 999px }
---

# Design System: càdlàg

## Overview

Retícula suiza de 12 columnas compuesta con asimetría y mucho aire (wabi-sabi). Papel con una fibra muy suave, filetes de 1px, un filete de 3px en tinta solo para lo principal (la matriz, los partidos decisivos). Una familia sans (Archivo) hace todo; su eje de ancho da carácter: ancha en el logotipo, angosta en los nombres de liga de las tarjetas.

## Rules

- **Un acento.** `beni` es el único color de marca. `ai` y `kaki` solo codifican zonas de la tabla y estados; nunca decoran.
- **Color con texto.** Todo color semántico va acompañado de su etiqueta en el lugar (p. ej. "● título 0,12").
- **Sin rótulos sobre cada sección.** Un h2 descriptivo basta. El texto pequeño en mayúsculas espaciadas queda solo para los encabezados de zona de la matriz.
- **Cifras grandes solo si son el dato.** Nada de filas de métricas decorativas.
- **Vidrio solo en barras fijas** (barra principal y barra de ligas): papel al 72 % con desenfoque.
- **Modo oscuro cálido.** Carbón `#2A2622`, nunca negro; la rampa de calor se invierte a coral sobre oscuro.
- **Movimiento con propósito.** Números que ruedan, celdas que se interpolan, filas que se reordenan (FLIP), mega menú escalonado, carrusel 3D (arrastre, flechas, teclado, trackpad lateral), línea càdlàg que se dibuja con el scroll solo en la portada. Todo se apaga con `prefers-reduced-motion`.
- **Logotipo.** El salto càdlàg: un tramo con círculo abierto al final y el siguiente con punto lleno al inicio, en `beni`.

## Components

- **Matriz:** celdas de 38px mínimo, porcentajes con una decimal, vacío como punto de 2px, cruz de resaltado y tooltip al pasar, columna de equipo fija.
- **Laboratorio:** columna lateral de 300px, sin caja; botones L/E/V cuyo ancho sigue la probabilidad; seleccionado en tinta.
- **Tarjeta de liga:** papel-2, borde `rule` (tinta cuando está activa), matriz en miniatura en canvas, favorito y su porcentaje abajo.
- **Botones:** píldora con borde tinta; relleno tinta para la acción principal.
