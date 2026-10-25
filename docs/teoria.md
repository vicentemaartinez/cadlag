# Notas teóricas

**Modelo oficial.** càdlàg usa en todas las ligas el modelo de dos latentes de §18: cada equipo
tiene fuerzas para crear y evitar tiros al arco y para convertirlos y evitar que se los conviertan.
Las secciones 1-8 desarrollan el núcleo de un latente (Dixon–Coles dinámico), del que el oficial
hereda el gauge, la actualización de Laplace, el ajuste prequential y la simulación; §18 indica
qué cambia. El núcleo se mantiene como modelo de referencia en los backtests y como respaldo
(§21).

Estas notas acompañan al código y serán la base de la Parte II del artículo. Cada resultado
se marca como **[lit]** (de la literatura), **[dem]** (demostrado aquí) o **[conj]** (conjetura
que el proyecto evalúa empíricamente).

## 1. Modelo

Para una liga con $N$ equipos, el estado es
$x_t=(\alpha_{1:N},\delta_{1:N},\mu,\eta)\in\mathbb R^{2N+2}$ y evoluciona en tiempo continuo:

$$
x_{t+\Delta}=x_t+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\Delta\,Q),\qquad
Q=\operatorname{diag}(\sigma_\text{eq}^2\mathbf 1_{2N},\sigma_\mu^2,\sigma_\eta^2),
$$

con un salto entre temporadas $(\alpha,\delta)\mapsto\phi(\alpha,\delta)+\mathcal N(0,\sigma_s^2 I)$
y un prior propio $\mathcal N((a_\text{asc},d_\text{asc}),v_\text{asc}I)$ para los ascendidos.
Un partido $i$ (local) vs. $j$ observa $(X,Y)\sim\mathrm{DC}(\lambda,\nu,\rho)$ con

$$
\log\lambda=\mu+\eta(1-\text{neutral})+\alpha_i-\delta_j,\qquad \log\nu=\mu+\alpha_j-\delta_i .
$$

## 2. Gauge

**Proposición [dem].** Sea $A$ el conjunto de equipos activos y
$g_a=(\mathbf 1_A,0,-1,0)$, $g_d=(0,\mathbf 1_A,1,0)$. Para todo partido entre equipos de $A$,
$(\log\lambda,\log\nu)$ es invariante bajo $x\mapsto x+a\,g_a+d\,g_d$.

*Demostración.* $\log\lambda$ cambia en $-a+d+a-d=0$, y lo mismo $\log\nu$. ∎

La verosimilitud es entonces plana en un subespacio de dimensión 2 y el posterior queda
determinado por el prior en esa dirección. Se fija el gauge con la proyección oblicua

$$
\Pi=I-G(C^\top G)^{-1}C^\top,\qquad G=[g_a\,g_d],\quad C=[(\mathbf 1_A,0,0,0)\;(0,\mathbf 1_A,0,0)],
$$

que satisface $\Pi^2=\Pi$, $C^\top\Pi=0$ (impone $\sum_A\alpha=\sum_A\delta=0$) y $\Pi g=0$ para
$g\in\operatorname{span}G$: cambia el representante sin cambiar ninguna predicción. A
diferencia de la proyección ortogonal, conserva la interpretación de $\mu$ como nivel de goles
del equipo promedio.

## 3. Actualización de Laplace en dos dimensiones

La verosimilitud depende de $x$ solo vía $z=Ax\in\mathbb R^2$. Si el prior es
$x\sim\mathcal N(m,P)$, entonces $z\sim\mathcal N(\bar z,S)$ con $\bar z=Am$, $S=APA^\top$, y

$$
p(x\mid\text{dato})\propto\mathcal N(x;m,P)\,L(Ax).
$$

**Lema [dem].** Si se reemplaza $p(z\mid\text{dato})$ por una gaussiana $\mathcal N(z^*,\Sigma^*)$,
la distribución de $x$ que conserva el condicional $x\mid z$ del prior es gaussiana con

$$
m'=m+K(z^*-\bar z),\qquad P'=P+K(\Sigma^*-S)K^\top,\qquad K=PA^\top S^{-1}.
$$

*Demostración.* Bajo el prior, $x\mid z\sim\mathcal N(m+K(z-\bar z),P-KSK^\top)$. Integrando
$z\sim\mathcal N(z^*,\Sigma^*)$: media $m+K(z^*-\bar z)$ y covarianza
$P-KSK^\top+K\Sigma^*K^\top$. ∎

Con $z^*=\arg\max\{\log\mathcal N(z;\bar z,S)+\log L(z)\}$ y
$\Sigma^*=(S^{-1}-\nabla^2\log L(z^*))^{-1}$ se obtiene el filtro de Laplace (filtro de
Kalman extendido iterado). Toda la no-gaussianidad se resuelve en un problema de Newton en
$\mathbb R^2$, y el costo por partido es $O(D^2)$ con $D=2N+2$.

## 4. Elo como caso límite [dem]

Tomemos Poisson sin $\tau$, un paso de Newton desde $\bar z$ y $S$ pequeño. Entonces
$z^*-\bar z\approx S\,\nabla\log L(\bar z)=S\,(x-\lambda,\;y-\nu)^\top$, y

$$
m'-m\approx PA^\top(x-\hat\lambda,\;y-\hat\nu)^\top .
$$

Para el ataque del local, con $P$ diagonal igual a $v I$, se obtiene
$\alpha_i'=\alpha_i+v\,(x-\hat\lambda)$: un **Elo de goles**, cuya constante $K=v$ es la
varianza del estado. En régimen estacionario, $v$ es el punto fijo de la ecuación de Riccati
$v=v+\sigma^2\Delta-v^2\hat\lambda/(1+v\hat\lambda)$, es decir
$v^*\approx\sqrt{\sigma^2\Delta/\hat\lambda}$. Así la regla ad hoc "elegir K" queda reemplazada
por una cantidad derivada: la razón entre la volatilidad de la fuerza y la información
que aporta un partido.

## 5. Ajuste de hiperparámetros

Con $\theta$ los 12 hiperparámetros, se maximiza

$$
\ell(\theta)=\sum_k\log p_\theta(x_k,y_k\mid\mathcal F_{k-1}),\qquad
\log p\approx -\tfrac12 (z^*-\bar z)^\top S^{-1}(z^*-\bar z)+\log L(z^*)+\tfrac12\log|\Sigma^*|-\tfrac12\log|S| ,
$$

(aproximación de Laplace de la predictiva), que es la regla logarítmica **prequential**
(Dawid, 1984) **[lit]**. El gradiente $\nabla_\theta\ell$ se obtiene por diferenciación
automática a través de las $\sim$6000 actualizaciones (incluidas las iteraciones de Newton).

## 6. Predictiva posterior y ley de varianza total

Para los puntos finales de un equipo, $\mathrm{Var}(\text{Pts}_i)=\mathbb E[\mathrm{Var}(\text{Pts}_i\mid x)]+\mathrm{Var}(\mathbb E[\text{Pts}_i\mid x])$.
El plug-in fija $x=m$ y elimina el segundo término. Si quedan $R$ partidos y la incertidumbre
de la fuerza induce una desviación $s$ en los puntos esperados por partido, ese término es
del orden de $R^2s^2$, mientras el primero crece como $R$. **[dem]** Por eso la diferencia entre
ambos modos es grande al comienzo de la temporada y se anula al final; el backtest de
temporadas lo mide (ver `outputs/backtest`).

Además, sortear el estado **una vez por temporada simulada** y no una vez por partido es
esencial: sortearlo por partido hace condicionalmente independientes los resultados de un
mismo equipo y equivale a un plug-in con las probabilidades predictivas.

## 7. Muestreo exacto de Dixon–Coles [dem]

Como $\mathrm{DC}=\tau\cdot(\mathrm{Pois}\otimes\mathrm{Pois})$ y $\tau\le\tau_{\max}$, aceptar una
propuesta Poisson-producto con probabilidad $\tau(x,y)/\tau_{\max}$ produce exactamente DC
(aceptación–rechazo con constante $\tau_{\max}$). La tasa de aceptación es $1/\tau_{\max}\ge 0{,}8$ para valores
típicos de $\rho$.

## 8. Puntuación

El RPS es propio; la distancia de Wasserstein-1 entre la distribución pronosticada y el
resultado observado, $\sum_k|F_k-\mathbb 1\{o\le k\}|$, no lo es: su valor esperado bajo $q$ es
$\sum_k[F_k(1-Q_k)+(1-F_k)Q_k]$, lineal en cada $F_k$, y se minimiza con $F_k\in\{0,1\}$ según
$Q_k\gtrless 1/2$, o sea, con toda la masa en la mediana de $q$. **[dem]**

---

# Extensiones (proyecto base completo)

## 9. Importancia de un partido como información mutua (E5)

Sea $O_k\in\{L,E,V\}$ el resultado de un partido pendiente y $F$ una variable de la tabla
final (el campeón $C$; el indicador de descenso $R_i$ de cada equipo). Se define

$$
\mathcal I_k^{\text{título}}=I(O_k;C),\qquad \mathcal I_k^{\text{descenso}}=\sum_i I(O_k;R_i),
$$

en bits. **Propiedades [dem]:** $0\le I(O_k;C)\le\min\{H(O_k),H(C)\}\le\log_2 3$; vale cero si y
solo si el resultado es independiente del campeón; es invariante a reetiquetar resultados.
Se estima con las mismas simulaciones de la temporada (cada simulación trae el resultado de
cada partido pendiente y la tabla final) y se corrige el sesgo del estimador *plug-in*
con Miller–Madow, $\hat I-(|O|-1)(|F|-1)/(2S\ln 2)$ **[lit]**. A diferencia de las medidas
clásicas (Schilling; Goossens et al.), no requiere elegir un umbral ni un evento: mide cuánta
incertidumbre sobre el desenlace resuelve el partido.

**¿Qué necesita mi equipo?** (i) *Curva probabilística*: $P(\text{objetivo}\mid \text{puntos que
sume})$, estimada por clases de las simulaciones; el "número mágico" probabilístico es el mínimo
de puntos con probabilidad $\ge 50\%$ o $\ge 90\%$. (ii) *Cotas matemáticas*: con 3-1-0, decidir
si un equipo puede salir campeón es NP-completo (Kern & Paulusma 2001; Bernholt et al.) **[lit]**,
así que la mejor y la peor posición posibles se calculan como programas enteros (HiGHS):

$$
\text{mejor}(i)=1+\min_{x}\ \#\{j:P_j(x)>P_i(x)\},\qquad \text{peor}(i)=1+\max_x\ \#\{j:P_j(x)\ge P_i(x)\},
$$

con $x$ las variables binarias de resultado de cada partido pendiente y los indicadores
linealizados con big-M. Los empates se tratan a favor/en contra porque la diferencia de gol no
está acotada por los puntos (cotas conservadoras).

**Suspenso.** Con la secuencia de creencias $\mu_t$ sobre el campeón a lo largo de la temporada,
la *sorpresa* de Ely–Frankel–Kamenica (2015) es $\sum_t\|\mu_t-\mu_{t-1}\|_2^2$ y la entropía
$H(\mu_t)$ mide la incertidumbre viva. Se reportan junto a medidas estáticas (HHI normalizado,
Gini, Noll–Scully con empates: la desviación de puntos por partido dividida por la de una liga
de equipos iguales con la misma tasa de empates).

## 10. Comparación entre ligas: la red eléctrica (E1)

**Modelo.** A las fuerzas relativas a cada liga se suman offsets de liga-temporada
$(A_{\ell,s},D_{\ell,s})$ con paseo aleatorio $\theta_{\ell,s}=\theta_{\ell,s-1}+\mathcal N(0,\tau^2I)$ y
referencia fija (Premier League). Observan los offsets: (a) los partidos de copa entre equipos
de ligas distintas (verosimilitud de Poisson con las fuerzas de cada filtro como *offset*);
(b) los equipos que cambian de división, como aristas gaussianas sobre la fuerza neta
$r=A+D$: $\text{net}_{i}^{(\ell,s)}+r_{\ell,s}\approx\text{net}_i^{(\ell',s+1)}+r_{\ell',s+1}$.
$\tau$ se elige maximizando la evidencia de Laplace.

**Proposición [dem].** En la aproximación de Laplace la precisión posterior de las fuerzas netas es
$H=L_w+L_{\text{tiempo}}+\Lambda_{\text{ref}}$, con $L_w=\sum_e w_e(\mathbf e_\ell-\mathbf e_{\ell'})(\mathbf e_\ell-\mathbf e_{\ell'})^\top$ el
laplaciano del grafo de ligas ponderado por la información de Fisher de cada partido o
cambio de división, $L_{\text{tiempo}}$ el laplaciano de la cadena temporal ($w=1/\tau^2$) y
$\Lambda_{\text{ref}}$ el anclaje (a tierra) de la referencia. Entonces

$$
\operatorname{Var}(r_\ell-r_{\ell'}\mid\text{datos})=(\mathbf e_\ell-\mathbf e_{\ell'})^\top H^{-1}(\mathbf e_\ell-\mathbf e_{\ell'})=R_{\text{eff}}(\ell,\ell'),
$$

la resistencia efectiva entre los nodos $\ell,\ell'$ de la red con conductancias $w$. Consecuencias:
agregar partidos es agregar resistencias en paralelo (la varianza solo baja, regla de Rayleigh);
la incertidumbre entre dos continentes está acotada inferiormente por la resistencia del "puente"
de partidos intercontinentales, por muchos partidos que haya dentro de cada continente.

**Espectro.** En el grafo de ligas (pesos = número de partidos y cambios de división, laplaciano
normalizado), el segundo autovalor es la conectividad algebraica (Fiedler) y su autovector
separa el grafo por su cuello de botella (desigualdad de Cheeger) **[lit]**.

## 11. Descomposición de Hodge (E3)

Ver docstring de `model/hodge.py`. En una liga todos-contra-todos el grafo es completo, el
complejo de cliques no tiene ciclos no triviales y la componente armónica es nula **[dem]**
($H^1=0$ para el 2-esqueleto de un símplice); toda la intransitividad es local (curl). El
nulo calibrado es un bootstrap paramétrico desde un modelo de fuerzas puras ajustado a cada
temporada. **Observación:** usar como nulo las predicciones un paso adelante del filtro es un
error, porque esas fuerzas están contraídas hacia la media (son estimaciones con incertidumbre),
lo que hace que las temporadas simuladas sean *menos* transitivas que la real.

## 12. Estilos latentes (E2)

Término bilineal $u_i^\top v_j$ sobre el predictor del filtro (ver `model/styles.py`).
**Proposición [dem].** El MAP de $(U,V)$ con prior $\mathcal N(0,s^2)$ resuelve
$\min_\Gamma\ \ell(\Gamma)+\|\Gamma\|_*/s^2$ (por la caracterización variacional de la norma
nuclear, $\|\Gamma\|_*=\min_{UV^\top=\Gamma}\tfrac12(\|U\|_F^2+\|V\|_F^2)$), y su solución es
$\Gamma=0$ si y solo si $\|\nabla\ell(0)\|_2\le 1/s^2$: un umbral sobre el mayor valor singular
de la matriz de residuos atacante × defensor. Esto convierte la pregunta "¿hay estilos?" en una
prueba espectral: comparar el mayor valor singular de la matriz de residuos (doblemente
centrada, para eliminar errores aditivos de fuerza) con su distribución bajo el modelo sin
interacción. Además, el MAP conjunto con escala jerárquica es degenerado (embudo de Neal:
la densidad diverge con $s\to0$, $U=V=0$), por lo que la escala se fija o se integra (NUTS).

## 13. Tiros al arco como segunda medición (E4)

Ver `model/filter_shots.py`. (Los números de R7 se recalcularon con tiros limpios: ver R7 y R13.) El punto de diseño: los tiros comparten con los goles solo la
parte de fuerzas $\alpha_i-\delta_j$; el nivel de la liga y la localía quedan determinados por los
goles. Compartir el predictor completo hace que la información (mucho mayor) de los tiros
distorsione $\mu$ y $\eta$ y empeora la predicción de goles: un ejemplo de mala especificación
amplificada por una observación muy informativa.

## 14. Modelo en vivo y contagio (E6)

Proceso de conteo con intensidad log-lineal por tramos y estados (marcador, rojas); la
verosimilitud es la de un GLM de Poisson sobre segmentos ("piecewise exponential").
**Resultado:** después de un gol, *ambos* equipos marcan menos durante 10 minutos (efecto
refractario: celebración, saque, reacomodo táctico), y el Hawkes aditivo estima excitación nula.
Los goles no se contagian: se inhiben. Las probabilidades en vivo resuelven las ecuaciones de
Kolmogorov hacia adelante $\dot p=pQ(\tau)$ con $Q$ constante por tramos (exponenciales de matriz).

## 15. Cambios de régimen (E7)

BOCPD de Adams–MacKay sobre los residuos de diferencia de gol de cada equipo, con prior
Normal–Gamma (predictiva t de Student). Nota: con tasa de riesgo constante,
$P(r_t=0\mid x_{1:t})=H$ siempre **[dem]** (el cambio en $t$ no es identificable con el dato de $t$),
así que la detección se hace con retraso: $P(r_{t+d}\le d\mid x_{1:t+d})$.

## 16. Validación del filtro contra NUTS

En una media temporada, el posterior exacto del estado (NUTS en NumPyro, paseo aleatorio
semanal, prior = estado del filtro al inicio) se compara con la gaussiana del filtro de
Laplace: medias, desviaciones y matrices de posiciones (`scripts/nuts_validation.py`).


# Extensiones de la segunda etapa (octubre de 2026)

## 17. Eventos raros: muestreo por importancia

Con $n$ temporadas simuladas el error relativo de Monte Carlo es $\sqrt{(1-p)/(np)}$: para
$p=10^{-6}$ y 10 % de error hacen falta $10^8$ temporadas. Un evento extremo de un equipo (que el
favorito descienda) ocurre combinando dos caminos: que el equipo *sea* peor de lo que se cree y
que le vaya mal. La propuesta inclina ambos:

* **fuerza**: el bloque $w=(\alpha_t,\delta_t)$ se sortea de $N(\mu_w-cu,P_{ww})$, con $u$ la
  dirección que baja $\alpha_t+\delta_t$ en una desviación a priori, y el resto del estado de su
  condicional exacta dado $w$;
* **resultados**: en cada partido del equipo su tasa se multiplica por $e^{-\theta}$ y la del rival
  por $e^{\theta}$.

La razón de verosimilitudes es exacta y se factoriza **[dem]**:
$$W=\frac{N(w;\mu_w,P_{ww})}{N(w;\mu_w-cu,P_{ww})}\prod_{m\ni t}
\frac{\mathrm{DC}(x_m,y_m;\lambda_m,\nu_m)}{\mathrm{DC}(x_m,y_m;\lambda_me^{-\theta},\nu_me^{\theta})},
\qquad \hat p=\tfrac1n\textstyle\sum_k W_k\mathbf 1\{A_k\},\quad E_q[\hat p]=p,$$
porque el resto de la simulación (paseo aleatorio, demás partidos, desempates) es idéntico bajo
ambas leyes. $(c,\theta)$ se eligen en corridas piloto minimizando la varianza relativa estimada
$E_q[W^2\mathbf 1_A]/E_q[W\mathbf 1_A]^2-1$.

*Lo que no funcionó:* inclinar solo el estado (entropía cruzada de Rubinstein, también con mezclas
del camino y covarianza adaptada) da estimadores de cola pesada para $p\sim10^{-7}$: tamaño
efectivo de 1 a 7 y semillas que discrepan en órdenes de magnitud. La mala suerte sin inclinar
casi nunca se muestrea, y es parte del camino dominante. Ver `sim/rare.py`, `scripts/rare_events.py`.

## 18. Modelo oficial: dos latentes para los tiros (creación y definición)

Cada equipo tiene $a_i,b_i$ (crear y evitar tiros al arco) y $f_i,g_i$ (convertirlos y evitar que
los recibidos sean gol):
$$\mathrm{SOT}_h\sim\mathrm{NB}(e^{c+k+a_i-b_j},r),\qquad X\mid \mathrm{SOT}_h\sim
\mathrm{Bin}\big(\mathrm{SOT}_h,\sigma(\phi+f_i-g_j)\big).$$
**Adelgazamiento [dem].** Si $S\mid\Lambda\sim\mathrm{Poisson}(\Lambda)$ con $\Lambda\sim
\mathrm{Gamma}(r, r/m)$ y $X\mid S\sim\mathrm{Bin}(S,p)$, entonces $X\mid\Lambda\sim
\mathrm{Poisson}(p\Lambda)$ y $p\Lambda\sim\mathrm{Gamma}(r,r/(pm))$: $X\sim\mathrm{NB}(pm,r)$. La
predictiva de goles queda en forma cerrada y los partidos sin estadísticas informan a través de
esa marginal. El estado vive en $\mathbb R^{4N+3}$, la verosimilitud depende de él por
$z\in\mathbb R^4$ (actualización de Laplace en cuatro dimensiones) y hay cuatro invarianzas de
gauge, fijadas con la misma proyección oblicua. Ver `model/filter_shots2.py`.

**Lo que hereda del núcleo y lo que cambia.** Cada bloque $(a,b)$ y $(f,g)$ sigue su propio paseo
aleatorio ($q_s$, $q_f$), con su propio salto entre temporadas ($\phi_s$, $\phi_f$, $s_s$, $s_f$). Un ascendido parte de un prior
propio en tiros ($a_\text{asc}$, $b_\text{asc}$, $v_\text{asc}$) y de definición promedio
($f=g=0$, varianza inicial $v_{0f}$); los niveles globales $c,k,\phi$ caminan con $q_\text{lev}$. Son 16
hiperparámetros, ajustados igual que en §5 pero con la log-verosimilitud predictiva **de los
goles** como objetivo (los tiros entran a la actualización, no al criterio): así el ajuste premia
pronosticar goles, que es lo que decide la tabla, y no explicar tiros. Un partido sin estadísticas
actualiza solo con la marginal $\mathrm{NB}(pm,r)$ de los goles. Los tiros al arco corruptos de
ESPN se descartan antes (`clean_sot`: total 0 o menos tiros al arco que goles − 1; temporadas
enteras fuera si más del 20 % de sus partidos fallan la regla). Para el sitio, ataque y defensa se
reportan en log-goles frente al promedio,
$\text{ataque}_i=a_i+\log\sigma(\phi+f_i)-\log\sigma(\phi)$ y análogo para la defensa, con su
desviación por el método delta: la misma escala que el núcleo.

## 19. Valor de mercado en la pretemporada

Con los traspasos de Transfermarkt de las cinco grandes ligas (github.com/eordo/transfermarkt-data,
2009-10 a 2025-26), para cada club y temporada
$x_i=(V^{\text{llega}}_i-V^{\text{sale}}_i)/\mathrm{mediana}_j(V^{\text{llega}}_j+V^{\text{sale}}_j)$
(valor de mercado de los jugadores, préstamos incluidos) y $y_i$ = fuerza neta filtrada al final
de la temporada menos la media a priori de pretemporada (lo que el modelo no anticipó, ya
descontada la regresión a la media). Se ajusta $y=\gamma x$ con temporadas anteriores a 2016 y se
evalúa fuera de muestra, también como desplazamiento $\gamma x_i$ de la media de pretemporada.

## 20. Traspasos como aristas del grafo entre ligas

Un traspaso definitivo entre ligas del grafo es una comparación por pares:
$P(A\to B)=\sigma\big(\beta(r_B-r_A)\big)$ sobre las mismas fuerzas netas $r$ de §10, con
cuasi-verosimilitud (dispersión $\kappa$ estimada por Pearson: los traspasos no son independientes
ni responden solo al nivel deportivo). Entran al mismo Newton/Laplace de §10 como un bloque
logístico. $\beta$ se elige por validación cruzada en cinco partes sobre los partidos de copa.
Interesa sobre todo porque los traspasos Sudamérica ↔ Europa cruzan el corte de Fiedler, donde
el grafo de partidos tiene casi nada.

## 21. Un solo modelo oficial

El núcleo (§1) y el de dos latentes (§18) comparten interfaz (`model/engine.py`): filtrar, pasar
de temporada, simular, resumir fuerzas. El oficial es el de dos latentes en todas las ligas. La
regla es asimétrica a propósito: una liga vuelve al núcleo solo si hay **evidencia de que el de
dos latentes pierde** (dos errores estándar en contra, en la log-verosimilitud del marcador o en la
de la posición final; `scripts/select_models.py` → `models/model_selection.json`). Elegir liga por
liga el que gane por dos errores estándar trataría como señal el ruido de siete pruebas: con la
evidencia combinada (R13, R16) lo razonable es "ayuda un poco en todas, más en algunas", no
"ayuda en tres y en cuatro no". Hoy ninguna liga cumple la condición para volver al núcleo.

## 22. Escenarios por reponderación ("¿y si…?")

Para cada temporada simulada $s$ se guardan las fuerzas implícitas de los próximos $K$ partidos
a través de $q_{sk}(o)=P(o_k=o\mid x_s)$, $o\in\{L,E,V\}$. Fijar resultados $o_k=\bar o_k$ para
$k\in F$ es condicionar. Como, dadas las fuerzas, los partidos son independientes,
$$P(\text{posiciones}\mid \bar o_F)\;\propto\;\mathbb E\Big[\mathbf 1\{\cdot\}\prod_{k\in F}
q_{sk}(\bar o_k)\Big],$$
es decir, cada temporada recibe peso $w_s=\prod_{k\in F}q_{sk}(\bar o_k)$ y en ella se reemplaza el
resultado sorteado por el fijado (ajustando puntos y reordenando). Es la versión Rao–Blackwell del
filtrado por rechazo (que descarta todas las temporadas donde el resultado sorteado no coincide):
usa todas las temporadas y su varianza es menor. Con $F$ grande el peso se concentra; el sitio
muestra el tamaño efectivo de muestra $(\sum w)^2/\sum w^2$ y avisa bajo 400. El reordenamiento
usa puntos y, en empates, el orden original de la simulación (aproximación a los desempates por
diferencia de goles, que no se re-sortean).

# Extensiones de la tercera etapa: ligas con ascenso y torneos

## 23. Championship: ascenso por playoffs, descendidos y descuentos

**Reglas.** 24 equipos, 46 fechas; ascienden directo los dos primeros y un tercero por playoffs;
descienden los tres últimos. Desempate: puntos, diferencia de gol, goles a favor, resultados entre
sí, victorias, goles de visita. Desde 2026-27 los playoffs son de seis equipos: 5.º-8.º y 6.º-7.º a
partido único en casa del mejor ubicado; el 3.º enfrenta al peor ubicado de los dos ganadores y el
4.º al otro, a ida y vuelta con la vuelta en casa del mejor ubicado; final única en Wembley. Hasta
2025-26 eran cuatro (3.º-6.º y 4.º-5.º a ida y vuelta), formato que se usa en los backtests. Las
llaves empatadas se definen con alargue y penales.

**Simulación.** En cada temporada simulada los playoffs se juegan con las fuerzas de esa misma
simulación al final del calendario (el estado $x_{T}$ del paseo aleatorio): la incertidumbre del
modelo sobre quién es mejor se propaga también a las llaves. El alargue se modela como un tercio de
partido con las mismas tasas, $\mathrm{NB}(\lambda/3, r)$, en la cancha del partido de vuelta (o
neutral), y los penales como una moneda. El resultado es, por simulación, la posición en la tabla
y si el equipo asciende; $P(\text{ascenso})=P(\text{1.º o 2.º})+P(\text{gana los playoffs})$.

**Descendidos frente a ascendidos [conj].** El prior de un equipo nuevo en la liga (§1, §18)
trataba igual a todos los que llegan. En la Championship llegan dos poblaciones muy distintas: los
tres descendidos de la Premier y los tres ascendidos de League One. En 2011-2025 los descendidos
promediaron 1,60 puntos por partido en su primera temporada, los ascendidos 1,22 y los que se
quedaron 1,35 (45, 45 y 270 temporadas-equipo). Por eso los que vienen de la división superior
(`League.parent`) reciben su propio prior en el bloque de tiros,
$(a,b)\sim\mathcal N((a_\text{desc},b_\text{desc}),v_\text{desc}I)$, con tres hiperparámetros más
que se ajustan con el resto (19 en total). Usar además la fuerza que el equipo tenía en la Premier
aporta poco: su correlación con los puntos por partido en la Championship es 0,20 (45 casos), que se
explica porque el descenso ya selecciona a los peores de la Premier. En ligas sin división
superior modelada la máscara de descendidos está vacía y los tres hiperparámetros no influyen.

**Descuentos de puntos.** La EFL descuenta puntos por administración o por infringir las reglas
financieras, y ESPN no los refleja en los resultados. Se cargan a mano con su fecha
(`leagues.POINT_ADJUSTMENTS`): la tabla y las simulaciones parten de ellos, y en los backtests un
pronóstico hecho antes del anuncio no los conoce (la tabla final, que es el blanco, sí).

**Laboratorio.** Con ascenso, el byte de posición de cada temporada simulada lleva un bit de
"asciende". Si los resultados fijados no cambian el orden de la tabla, se usa ese bit; si lo
cambian, el ascenso es directo en los dos primeros puestos y, en la zona de playoffs, se reemplaza
por $q_{t,k}=P(\text{gana los playoffs}\mid\text{equipo }t,\text{ posición }k)$, estimada con
todas las temporadas simuladas (no hace falta volver a jugar las llaves en el navegador).

## 24. Argentina: dos torneos, zonas, playoffs y promedios

**Formato (desde 2025).** Un año tiene dos torneos, Apertura y Clausura. 30 equipos en dos zonas
de 15: 14 partidos de zona y 2 interzonales (el clásico y uno sorteado), 16 fechas; el Clausura
repite el fixture del Apertura con las localías invertidas. Los 8 primeros de cada zona juegan
octavos (1.ºA-8.ºB, 1.ºB-8.ºA, 2.ºA-7.ºB, …), cuartos y semis a partido único en casa del mejor
ubicado y final neutral, con alargue y penales. La tabla anual suma la fase regular de los dos
torneos.

**Descensos.** Descienden dos: el último de los promedios y el último de la tabla anual; si son el
mismo equipo, el penúltimo de la tabla anual. El promedio de $i$ es
$$\pi_i=\frac{P_i^{(t-2)}+P_i^{(t-1)}+P_i^{(t)}}{J_i^{(t-2)}+J_i^{(t-1)}+J_i^{(t)}},$$
puntos y partidos de fase regular de las tres últimas temporadas en Primera (las que tenga). Los
dos años anteriores entran como constantes (`leagues.PROMEDIO_BASE`, desde la tabla oficial: ESPN
no trae la Copa de la Liga 2024) y el año en curso sale de cada temporada simulada, con el número
final de partidos conocido. Empates en el último lugar de la tabla anual: partido de desempate
neutral.

**Simulación.** Lo que cambia respecto de las demás ligas está en `sim/split.py`. (i) La zona de
cada equipo se infiere del fixture: dos equipos de la misma zona comparten casi todos sus rivales,
dos de zonas distintas casi ninguno. (ii) ESPN publica el Clausura por tramos; los partidos que
faltan se completan con el Apertura invertido y no se presentan como partidos reales. (iii) En
cada temporada simulada se arman las tablas de zona del torneo en curso, se juega la llave con las
fuerzas de esa simulación (los cruces ya jugados quedan fijos con su ganador real) y se calculan
promedios y tabla anual. La matriz del sitio es la de la tabla anual; campeón, clasificación a
playoffs y descenso van como columnas aparte. En el laboratorio, si los resultados fijados cambian
el orden de una zona, el campeón se reemplaza por $P(\text{campeón}\mid\text{equipo},\text{puesto en la
zona})$, estimada con todas las temporadas simuladas, y el descenso se recalcula exacto con los
promedios.

## 25. "¿Qué necesita?": la curva de puntos completa

La ficha de cada equipo muestra $p(g)=P(\text{objetivo}\mid\text{suma }g\text{ puntos desde hoy})$ y
los $g$ con $p\ge 0{,}5$ y $p\ge 0{,}9$. Contar éxitos entre las temporadas simuladas en que el equipo
sumó exactamente $g$ es exacto, pero solo donde hay simulaciones: un equipo muy superior casi nunca
suma pocos puntos y uno modesto casi nunca gana todo (con la regla "mínimo $g$ con 30 simulaciones",
el Real Madrid "necesitaba" 41 puntos para salvarse y el Betis "no salía campeón ni ganando todo").
Donde hay pocas, además, la curva sale con saltos que son ruido de Monte Carlo, no del fútbol.

**Remuestreo condicional [dem].** Sean $r=(r_1,\dots,r_K)$ los resultados de los $K$ partidos que le
quedan al equipo y $x$ las fuerzas de una temporada simulada. Dadas las fuerzas, los partidos son
independientes, así que el resto de esa temporada sigue siendo una muestra válida si se cambia $r$.
Para la familia inclinada
$$q_\theta(r\mid x)\propto p(r\mid x)\,e^{\theta\,g(r)},\qquad g(r)=\text{puntos del equipo},$$
se cumple $q_\theta(r\mid g(r)=g,x)=p(r\mid g(r)=g,x)$: la inclinación depende solo de $g$, así que
condicionada a un total deja intacta la distribución de *qué* partidos gana. En cada temporada
simulada se vuelven a sortear solo esos $K$ resultados, para una grilla de $\theta$ que lleva el
total desde casi cero hasta casi todos los puntos; los rivales ganan o pierden esos mismos partidos
y la tabla se reordena (desempate: el orden original de esa simulación). Dentro de cada $g$ todas
las muestras valen lo mismo y se juntan entre $\theta$: hay miles en cada punto de la curva. Lo único
que no se ajusta es la fuerza del propio equipo dado $g$, que solo pesa por su correlación a
posteriori con la de los demás. Se impone además que la curva no baje con $g$ (regresión
isotónica). En la Championship el ascenso por playoffs, y en Argentina ganar el torneo, se toman de
$P(\text{lo logra}\mid\text{equipo},\text{puesto})$ estimada con todas las simulaciones; el descenso
argentino recalcula promedios y tabla anual en cada muestra.

Con esto, los puntos finales que dan 50 % de salvarse coinciden entre los equipos de cada liga
(Premier 38-39, LaLiga 36-38, Serie A 35-36, Bundesliga 30-32, Ligue 1 30-31, Championship 46-48);
los del título difieren con razón, porque dependen de a quién hay que superar (al Bayern le basta
70, a los demás 82-83). Los escalones que quedan en la curva del título son reales: los rivales
suman de a 3 y de a 1.

# Extensiones de la cuarta etapa: copas

## 26. Champions League: una escala común para toda Europa

En una copa se enfrentan equipos de ligas distintas, y 15 de los 36 de la Champions 2026-27 juegan
en ligas que no modelamos (Portugal, Países Bajos, Turquía, Bélgica, Grecia, Chequia, Austria,
Noruega, Ucrania, Eslovaquia, Azerbaiyán). Hace falta una fuerza de cada equipo en una **escala
común**, con su incertidumbre.

**Capa europea.** Cada partido UEFA (Champions, Europa League, Conference y, hasta 2019, sus
previas; 9.700 desde 2010) entre $i$ (local) y $j$ observa
$$\log E[X]=\mu+\eta(1-\text{neutral})+\text{att}_i-\text{def}_j,\qquad X\sim\text{NegBin}(\cdot,r),$$
y lo simétrico para la visita. Para un equipo de una liga modelada $\ell$,
$$\text{att}_i=\gamma\,\text{att}^{\text{dom}}_i+A_\ell+u_i,\qquad \text{def}_i=\gamma\,\text{def}^{\text{dom}}_i+D_\ell+v_i,$$
donde la fuerza doméstica sale del modelo oficial linealizado en goles: como
$\log E[\text{goles}]=c+a_i-b_j+\log\sigma(\phi+f_i-g_j)\approx\text{cte}+a_i-b_j+\kappa(f_i-g_j)$ con
$\kappa=1-\sigma(\phi)$, es $\text{att}^{\text{dom}}=a+\kappa f$ y $\text{def}^{\text{dom}}=b+\kappa g$, con la media y la
covarianza del filtro doméstico en la fecha del partido. $(A_\ell,D_\ell)$ son los offsets de la liga
(la Premier es la referencia) y $(u_i,v_i)\sim N(0,v_u)$ un **residuo europeo**: lo que el equipo
rinde en Europa por encima o por debajo de lo que dice su liga (rota en la liga, juega distinto en
eliminatorias); revierte a 0 entre temporadas. Para un equipo sin liga modelada, $(\text{att}_i,\text{def}_i)$
es un estado propio con prior $N(\beta_c,v_0)$ al aparecer, según la copa que juega
($c\in\{$Champions, Europa League, Conference, previas$\}$: clasificar a la Champions ya dice algo
de la fuerza), paseo aleatorio dentro de la temporada y reversión hacia $\beta_c$ entre temporadas,
$m\leftarrow\beta_c+\varphi^k(m-\beta_c)$, $P\leftarrow\varphi^{2k}P+s^2(1-\varphi^{2k})/(1-\varphi^2)$ tras $k$
temporadas sin jugar. Los parámetros globales $(\mu,\eta,A_\ell,D_\ell)$ tienen un paseo aleatorio
lento y los offsets reciben varianza $\tau^2$ entre temporadas.

**Filtro factorizado.** Con 590 equipos, la covarianza conjunta no cabe en el gradiente. Se usa un
filtro de densidad supuesta (ADF) factorizado: el bloque global con su covarianza completa y un
bloque $2\times2$ por equipo, independientes entre sí. En cada partido se juntan los bloques que
intervienen (dimensión 14), se hace la actualización de Laplace de §3 exacta en ese estado local, y
cada bloque se queda con su marginal. Se pierde la correlación que un partido crea entre los dos
equipos (es la aproximación de TrueSkill); a cambio cada paso cuesta $O(1)$ y el filtro completo
corre en dos segundos, de modo que los 19 hiperparámetros se ajustan como en §5, por
verosimilitud prequential con gradientes exactos (scan con *checkpointing* por bloques de 128
partidos para que el gradiente quepa en memoria).

**Simulación.** Cada edición simulada sortea una vez los parámetros globales, la fuerza doméstica
y el residuo de cada equipo; con eso se juegan los partidos de la fase liga que faltan, se ordena
la tabla con los desempates del reglamento (puntos, diferencia, goles, goles de visita, victorias,
victorias de visita, puntos, diferencia y goles de los rivales; fair play y coeficiente no se
modelan) y se juega el cuadro: playoff 9-24 con cruces fijos (9/10 contra 23/24, …), octavos contra
el top 8, cuartos y semis fijos dentro de cada mitad, vuelta en cancha del puesto mejor sembrado,
alargue con tasas de un tercio y penales como moneda. Lo sorteable (qué 9.º enfrenta a qué 23.º, en
qué mitad queda cada par de sembrados) se sortea en cada simulación hasta que los cruces aparecen
en los datos; con los cruces publicados se reconstruye el cuadro. La final de 2027 es en el
Metropolitano: si llega el Atlético, juega de local.

El modelo de una copa con fase de grupos (hasta 2023-24, `sim.ucl_groups`) se implementó solo para
validar con más ediciones.

---

# Resultados empíricos del proyecto base (octubre de 2026)

Todos los números salen de `outputs/` y se regeneran con los scripts indicados.

| # | Pregunta | Resultado | Script |
|---|---|---|---|
| R1 | ¿Qué tan bueno es el núcleo partido a partido? | RPS 0,1995 vs. Bet365 0,1941 (top 5, 2016-26, 18.700 partidos); Brasil 0,2065 vs. 0,1999; Chile 0,2241 vs. base 0,2306 | `backtest.py` |
| R2 | ¿Plug-in vs. predictiva posterior en temporadas? | Varianza de puntos subestimada ×2,03 por el plug-in en pretemporada (cobertura 90 %: 76 %); la posterior baja a ×1,28 (85 %). Log score de posición mejor en todas las fracciones | `backtest.py` |
| R3 | ¿Es exacto el filtro de Laplace? | Contra NUTS (Premier 2025-26, mitad de temporada): correlación de medias 0,9999, diferencia máx. 0,015, razón de desviaciones 0,96, distancia de variación total en posiciones 0,02 (ruido Monte Carlo); 0 divergencias | `nuts_validation.py` |
| R4 | ¿Hay intransitividad (curl) más allá de las fuerzas? | No: 108 temporadas, proporción cíclica observada 0,596 vs. 0,590 del nulo; Fisher p = 0,81. Chile es la liga más "cíclica" (0,72) pero dentro de su nulo | `hodge_analysis.py` |
| R5 | ¿Hay estilos latentes (bajo rango)? | No: el mayor valor singular de los residuos doblemente centrados (8,55) queda bajo el percentil 95 del nulo (9,66) en todas las ligas; con s = 0,2 el MAP es nulo (umbral nuclear) y con s ≥ 0,5 sobreajusta (−38 a −68 milinats/partido) | `styles_analysis.py` |
| R6 | ¿Comparación entre ligas? | 2.208 partidos de copa + 611 cambios de división. λ₂ = 0,017 frente a λ₃ = 0,61: el vector de Fiedler separa Europa de Sudamérica. R_eff dentro de Europa ≈ 0,02-0,05; Europa–Sudamérica ≈ 0,10-0,16. Offsets netos vs. Premier: LaLiga −0,09, Serie A −0,13, Bundesliga −0,13, Ligue 1 −0,18, Brasileirão −0,18 ± 0,32, Argentina −0,49 ± 0,35, Chile −0,85 ± 0,37 | `crossleague_analysis.py` |
| R7 | ¿Los tiros al arco mejoran la predicción de goles? | **Corregido en la segunda etapa.** La versión original ("no ayudan") estaba contaminada por tiros al arco corruptos de ESPN (Premier 2016-17 entera con SOT ≈ 0, justo en el período de prueba). Con datos limpios, el modelo de un solo latente sí ayuda en la Premier: +0,0057 ± 0,0026 nats por partido, RPS 0,1988 frente a 0,1999, con acople k₁ = 0,90. El de dos latentes (R13) ayuda más | `shots_analysis.py` |
| R8 | ¿Efectos de estado en vivo? | Roja propia ×0,60 y del rival ×1,49 sobre la tasa de goles; ir perdiendo por 1 ×1,12; descuento del 1.er tiempo ≈ 2,3 minutos-equivalentes y del 2.º ≈ 5,5 | `inplay_analysis.py` |
| R9 | ¿Contagio (Hawkes)? | Al revés: en los 10' siguientes a un gol, el que anotó marca ×0,86 y el que recibió ×0,81 (LR = 363); excitación aditiva estimada en 0. Proceso refractario, no autoexcitado | `inplay_analysis.py` |
| R10 | ¿El filtro llega tarde a los cambios de régimen? | No: tras 132 detecciones BOCPD, el residuo medio de los 10 partidos siguientes es 0,025 ± 0,046 | `changepoint_analysis.py` |
| R11 | Competitividad | Entropía del campeón en pretemporada: Brasil 3,39 y Chile 3,32 bits; Bundesliga 1,23. La Bundesliga y la Ligue 1 se definen (P ≥ 95 %) al 44-46 % de la temporada; Chile y Brasil al 91-93 %. Campeones más improbables: Lille 2020-21 (1,2 %), Monaco 2016-17 (1,3 %) | `suspense_analysis.py` |

## Lectura de conjunto

Las ideas más atractivas a priori (estilos, intransitividad, contagio) dan resultados
**negativos limpios** (los tiros, que estaban en esta lista, resultaron positivos al limpiar los datos: R7, R13), con pruebas bien calibradas. Que el
nulo de Hodge con predicciones un paso adelante saliera *al revés* (realidad más transitiva que
el nulo) fue la pista de que esas predicciones están contraídas; corregirlo es parte del
resultado. Lo que sí funciona con fuerza es la propagación honesta de la incertidumbre (R2, R3)
y la geometría del grafo entre ligas (R6), que es el aporte más original.

## Pendientes formulados

Los pendientes 1 a 4 de la primera etapa están implementados (§17-20, resultados R12-R15); el
modelo de dos latentes es el oficial en todas las ligas (§21, R16) y el laboratorio del sitio
condiciona por reponderación (§22). La Championship (§23, R17-R18), Argentina (§24, R19) y la Champions League (§26, R20) están
implementadas. Quedan: las otras copas (Libertadores, Europa League) y, para la Champions, las ligas
domésticas de los equipos que hoy se estiman solo con partidos europeos.

## Resultados de la segunda etapa

| # | Pregunta | Resultado | Script |
|---|---|---|---|
| R12 | ¿Se pueden estimar probabilidades de 10⁻⁶? | Sí, inclinando fuerza **y** resultados del equipo (§17). Validado contra 600.000 temporadas directas: "el City termina 10.º o peor" 2,41·10⁻⁴ ± 5 % frente a 2,48·10⁻⁴ ± 8 %, con 15 veces menos simulación por igual error. Con semillas consistentes: el City desciende ≈ 2·10⁻⁷ (1 en 5 millones), el PSG ≈ 7·10⁻⁷, Gladbach (último) campeón ≈ 8·10⁻⁷, Bournemouth (último) campeón ≈ 1,2·10⁻⁴. Bajo ~10⁻⁸ las semillas discrepan (Barcelona, Bayern, Inter descienden: solo orden de magnitud, 10⁻⁹ a 10⁻¹³). En Chile y Brasil, con la liga avanzada, los cuatro eventos son imposibles por la cota entera | `rare_events.py`, `rare_refine.py` |
| R13 | ¿Separar creación y definición mejora la predicción de goles? | **Sí, en el conjunto de las ligas.** Fuera de muestra (desde 2016), log-verosimilitud del marcador por partido frente al núcleo: Premier +0,0093 ± 0,0023 (RPS 0,1982 frente a 0,1999; Bet365 0,1936), Chile +0,0042 ± 0,0017 (0,2219 frente a 0,2232), Ligue 1 +0,0052 ± 0,0024 (0,2048 frente a 0,2054); Brasileirão +0,0025 ± 0,0015 (no alcanza dos errores estándar), Serie A +0,0016 ± 0,0023, LaLiga +0,0004 ± 0,0015 y Bundesliga −0,0017 ± 0,0024, neutros. Ninguna liga empeora de forma significativa. Combinando las siete (promedio ponderado por la inversa de la varianza): **+0,0028 ± 0,0007** (z = 3,9), con heterogeneidad moderada (Q = 15,8 con 6 g. l.; la Premier es la que más gana); con la Championship (R17), las ocho: +0,0036 ± 0,0007 (z = 5,6); con Argentina (R19), las nueve: +0,0036 ± 0,0006 (z = 5,9). La definición es una habilidad persistente, no ruido: desviación entre equipos 0,05 a 0,11 en escala logit, con φ ≈ 0,76-0,90 entre temporadas; la del arquero es menor (0,03-0,09). Es el modelo oficial en las siete ligas (§21). Dato corregido en el camino: ESPN trae tiros al arco corruptos en temporadas enteras (Premier 2011-12, 2016-17 y 2022-23, con SOT ≈ 0); se descartan con una regla de consistencia y R7 se recalcula con datos limpios | `shots2_analysis.py`, `select_models.py` |
| R14 | ¿El balance de traspasos mejora la pretemporada? | No. γ = 0,04 ± 0,01 dentro de muestra, pero fuera de muestra (2016-25) la correlación es −0,04 y el pronóstico de pretemporada empeora levemente (log score de posición +0,004 ± 0,004 en 50 temporadas). Para ascendidos γ = 0,11, pero aplicarlo solo a ellos es neutro (+0,0003 ± 0,005). La regresión a la media del filtro ya captura lo que el mercado anticipa | `market_prior_analysis.py` |
| R15 | ¿Los traspasos sirven como aristas entre ligas? | No: el mercado mide dinero, no nivel. Con 8.377 traspasos definitivos entre ligas del grafo, la jerarquía que implican (Bradley–Terry) no se correlaciona con la de los partidos (Spearman −0,02: la Serie B y LaLiga 2 "atraen" más que la Premier, la Ligue 1 "exporta"). La validación cruzada en partidos de copa elige β = 0 y empeora de forma monótona al subirlo (−6265 → −6281). Forzarlos (β = 2, κ = 2,3) bajaría la resistencia Inglaterra–Chile de 0,134 a 0,050: precisión falsa | `transfer_edges_analysis.py` |
| R16 | ¿Y a nivel de temporada, que es lo que muestra el sitio? | Backtest pareado de la posición final real (−log P), desde 2016 (Chile desde 2020), en pretemporada y al 25, 50 y 75 % de la temporada, con hiperparámetros fuera de muestra: dos latentes **+0,0047 ± 0,0036** nats por equipo y temporada a favor (64 temporadas; error agrupado por temporada); con la Championship (R17), +0,0074 ± 0,0037 en 74; con Argentina (R19), +0,0075 ± 0,0034 en 84. Por liga: Premier +0,017 ± 0,009, Chile +0,013 ± 0,007, Brasileirão +0,012 ± 0,009, Ligue 1 +0,003 ± 0,011, Bundesliga +0,001 ± 0,010, LaLiga −0,001 ± 0,009, Serie A −0,009 ± 0,009: ninguna con evidencia de que pierda. La ganancia está en la segunda mitad (+0,008 al 50 %, +0,012 al 75 %); en pretemporada es nula, como se espera: ahí manda la regresión a la media entre temporadas, no la conversión. Frente al mercado, en los mismos 17.910 partidos del top 5 con cuotas: RPS 0,1990 (núcleo 0,1995, Bet365 0,1941); Premier 0,1977 (núcleo 0,1994, Bet365 0,1936) | `season_compare.py`, `shots2_vs_market.py` |
| R17 | ¿Funciona en una liga con ascenso por playoffs (Championship)? | Sí. Partidos, desde 2016-17 (5.689): dos latentes +0,0075 ± 0,0015 nats frente al núcleo (z = 4,9); en los 5.552 con cuotas, RPS 0,2183 frente a 0,2203 del núcleo y 0,2142 de Bet365. Temporadas (10, formato de cuatro equipos en playoffs): posición final +0,025 ± 0,014; ascenso (−log P de subir o no, con el ascenso real leído de la Premier del año siguiente) +0,0095 ± 0,0052 por equipo y momento. El ascenso está calibrado: la suma de probabilidades da 3,0 ascensos por temporada en cada momento, y por tramos el pronóstico y la frecuencia coinciden (0,09 frente a 0,07; 0,22 frente a 0,21; 0,37 frente a 0,43; 0,60 frente a 0,67; 0,87 frente a 0,89). Brier del ascenso en pretemporada 0,094 (núcleo 0,100) | `backtest.py eng.2 2016`, `shots2_analysis.py`, `season_compare.py`, `shots2_vs_market.py` |
| R18 | ¿Sirve el prior propio para los descendidos (§23)? | Sí. Frente al mismo modelo sin él: +0,0019 ± 0,0005 nats por partido (z = 3,6), +0,0038 ± 0,0012 en los partidos de equipos recién llegados; en el ascenso +0,0046 ± 0,0023, casi todo en pretemporada (Brier 0,094 frente a 0,099). El ajuste fuera de muestra separa las dos poblaciones en el bloque de tiros: descendidos $a=+0,10$, $b=+0,13$; ascendidos $a=0,00$, $b=-0,05$ | `shots2_norel_fit.py`, `season_compare.py` |
| R19 | ¿Y en Argentina (zonas, playoffs, promedios)? | Partidos desde 2016 (4.218): dos latentes +0,0035 ± 0,0019 frente al núcleo (z = 1,9, sin evidencia de que pierda, así que queda el oficial); en los 1.944 con cuotas, RPS 0,2134 frente a 0,2140 del núcleo y 0,2071 de Bet365. Temporadas (10, tabla de cada año): +0,008 ± 0,009. Torneos con el formato actual (Apertura y Clausura 2025, Apertura 2026), fuera de muestra: la clasificación a octavos se pronostica bien (Brier 0,12 promediando momentos, frente a 0,25 de repartir 16 de 30 al azar; 0,118 el oficial y 0,120 el núcleo), pero los tres campeones fueron sorpresas: Platense, Estudiantes y Belgrano tenían entre 1,4 y 6,3 % en todos los momentos, con favoritos de 11-23 %. Una llave de 16 a partido único con alargue y penales reparte mucho; tres torneos no alcanzan para evaluar la calibración del campeón | `backtest.py arg.1 2016`, `shots2_analysis.py`, `season_compare.py`, `tournament_backtest.py` |
| R20 | ¿Sirve anclar los equipos europeos a su liga para pronosticar la Champions (§26)? | **Sí.** Partidos UEFA de 2018-19 a 2026-27 (4.046, hiperparámetros ajustados con 2011-2017): la capa con fuerzas domésticas gana **+0,018 ± 0,003** nats por partido (z = 5,6) a la misma capa solo con partidos de copa; la ganancia está en los partidos entre un equipo de las cinco grandes ligas y uno de otra (Europa League +0,061 ± 0,012). El residuo europeo agrega +0,0016 ± 0,0007. Torneos (ocho ediciones, 2018-19 a 2025-26, en ocho o nueve momentos de cada una): log score de la etapa alcanzada **−0,020 ± 0,008** por equipo a favor (siete de ocho ediciones); el del campeón, −0,11 ± 0,18 (dos de las ocho las ganó el PSG, que al empezar tenía 1,9 % y 5,6 % con la capa y 4,7 % y 10,7 % solo con copas: la Ligue 1 pesa en su contra). Offsets netos frente a la Premier en 2026: LaLiga −0,30, Bundesliga −0,32, Serie A −0,39, Ligue 1 −0,51. Probado y descartado: una media propia para los equipos presentes al empezar los datos y una media común de reversión (mejor dentro de muestra, peor fuera: 2,989 frente a 2,974) | `europe_fit.py`, `ucl_backtest.py` |

## Lectura de la segunda etapa

Dos resultados positivos y dos negativos, todos con validación fuera de muestra. Lo que funciona
es **más estructura en el modelo de goles**: separar crear ocasiones de convertirlas (R13) mejora
la predicción de partidos en el conjunto de las ligas (z = 3,9 en las siete primeras, 5,6 con la Championship) y no empeora ni partidos ni
temporadas en ninguna (R16), así que pasa a ser el modelo oficial. Lo que agrega es justamente no
confundir una racha de conversión con fuerza: un equipo que convierte la mitad de sus tiros al arco
baja, uno al que le convierten casi cada tiro en contra sube. También funciona un **mejor Monte
Carlo** (R12: probabilidades de una en millones con un error controlado y validado
contra fuerza bruta). Lo que no funciona es **información de mercado**: el balance de traspasos
no anticipa nada que el filtro no sepa ya (R14), y la dirección de los traspasos refleja dinero,
no nivel deportivo (R15). La lección de método vino de un error de datos: R7 decía que los tiros
no servían porque una temporada entera de la Premier venía con ceros; revisar la consistencia de
los datos antes de concluir un negativo cambió la conclusión.
