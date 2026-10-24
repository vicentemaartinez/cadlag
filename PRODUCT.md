# Product

## Platform

web

## Users

Hinchas curiosos y gente de datos que quieren saber cómo termina su liga y qué necesita su equipo. Llegan desde un link compartido o buscando "probabilidades Colo Colo campeón". La mayoría no sabe qué es un filtro de Laplace ni le importa; una minoría (estudiantes, analistas) sí quiere el método.

## Product Purpose

càdlàg muestra la probabilidad de que cada equipo termine en cada posición de siete ligas (Premier League, LaLiga, Serie A, Bundesliga, Ligue 1, Liga de Primera de Chile, Brasileirão), simulando 12.000 veces lo que queda de cada temporada con las reglas reales de desempate. Éxito: (1) un hincha entiende en segundos dónde termina su equipo y qué partidos importan; (2) alguien de matemáticas encuentra un método serio y evaluado.

## Positioning

Una matriz de posiciones al estilo Opta, pero con el método a la vista y evaluado contra el mercado. Se diferencia por el laboratorio de escenarios (condicionar en resultados futuros sin re-simular), "qué necesita mi equipo" y la comparación entre ligas a través de las copas.

## Operating Context

El motor en Python actualiza datos de ESPN y simula por liga; la web lee JSON estáticos. Actualización manual o diaria.

## Capabilities and Constraints

- Matriz de posiciones, partidos decisivos (información mutua), laboratorio ¿y si…?, cajón de equipo (distribución, fuerzas, curva de puntos necesarios, cotas por programación entera), comparación entre ligas.
- Las probabilidades son de simulación Monte Carlo: el laboratorio filtra muestras, con aviso bajo 400 temporadas compatibles.
- No inventar resultados, métricas ni afirmaciones que no estén en `docs/teoria.md` (tabla R1–R11).
- Sin apuestas: las cuotas se usan solo para evaluar.

## Brand Commitments

Preciso, sobrio, con humor seco. El nombre viene de las trayectorias càdlàg (continua por la derecha, con límite por la izquierda): las probabilidades quedan quietas entre partidos y saltan con cada resultado.

Evitar:
- Párrafos explicativos donde un rótulo o una leyenda basta. Los números y gráficos se explican solos o con una línea.
- Etiquetas en mayúsculas espaciadas sobre cada sección; filas de cifras gigantes como adorno; titulares ingeniosos donde basta uno descriptivo.
- Estética de casa de apuestas: verdes neón, contadores de cuotas, urgencia.
- Bitácoras de "terminal" falsas y jerga técnica decorativa.

## Product Principles

1. **La matriz manda.** Todo lo demás la acompaña y no compite con ella.
2. **Mostrar, no explicar.** Una leyenda en el lugar del dato vale más que un párrafo arriba.
3. **Honestidad estadística.** Mostrar incertidumbre (intervalos, tamaño de muestra) y también lo que no funcionó.
4. **Dos niveles.** El hincha lee los porcentajes; el curioso encuentra el método un clic más abajo.

## Accessibility & Inclusion

WCAG 2.1 AA, foco visible, `prefers-reduced-motion` respetado, colores semánticos nunca como único canal (cada color va con texto). Español de Chile, números con formato es-CL.
