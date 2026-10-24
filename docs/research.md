# Un modelo latente dinámico genera la tabla final

La base matemática más sólida para este proyecto es un **modelo bayesiano de espacio de estados para goles**. Cada equipo tiene fuerzas de ataque y defensa que evolucionan como paseo aleatorio dentro de la temporada y con reversión a la media entre temporadas. Los goles siguen una verosimilitud de conteo con dependencia de bajo marcador (Dixon–Coles o Poisson bivariada). Los efectos de liga son jerárquicos y se identifican gracias a partidos internacionales, transferencias y valores de mercado. Sobre ese núcleo se ejecuta una **simulación Monte Carlo predictiva posterior** que aplica las reglas reales de desempate y play-off de cada liga y produce la matriz de posiciones finales al estilo Opta. La elección del modelo de goles puede guiarse por la elegancia sin pagar casi nada en precisión. En el Soccer Prediction Challenge 2017, el ganador oficial obtuvo un RPS de **0,2063** frente a **0,2255** de una línea base ingenua, y los cuatro primeros quedaron separados por apenas **0,003** ([Berrar et al. 2019](https://push-zb.helmholtz-munich.de/deliver.php?id=22296)). Los aportes con más potencial de originalidad son cinco: una teoría de identificabilidad de la comparación entre ligas basada en la **resistencia efectiva del grafo de partidos**; un término bilineal de "estilos" que induce un modelo intransitivo tipo *blade-chest*; una descomposición de Hodge calibrada contra un nulo; la importancia de un partido medida como **información mutua**; y la verificación de la calibración de las propias matrices de posición. La evaluación debe apoyarse en reglas de puntuación propias: log-score como métrica principal, RPS como secundaria, diagramas CORP, validación *leave-future-out* y un benchmark de mercado sin margen (Shin). El año 2026 trajo cuatro advertencias prácticas: FBref perdió el xG de Opta, la API de ClubElo cayó, *transfermarkt-datasets* se congeló y las cuotas de Pinnacle desaparecieron del CSV 2026/27 de football-data.co.uk. Chile es, por lejos, la liga con peores datos.

## Los modelos de goles difieren poco en precisión y mucho en estructura

El punto de partida es el modelo de Maher (1982): goles locales y visitantes independientes y Poisson, con predictor log-lineal ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)). Con local $i$, visitante $j$ y partido $k$:

$$
x_k\sim\mathrm{Pois}(\lambda_k),\quad y_k\sim\mathrm{Pois}(\mu_k),\qquad
\log\lambda_k=\mu_0+\eta+\alpha_i-\delta_j,\qquad \log\mu_k=\mu_0+\alpha_j-\delta_i .
$$

Aquí $\alpha$ es ataque, $\delta$ es defensa (mayor valor significa mejor defensa) y $\eta$ es la localía. El predictor es invariante bajo $\alpha_i\mapsto\alpha_i+c$, $\delta_i\mapsto\delta_i+c$, así que hace falta **una restricción de suma cero por instante**, $\sum_i\alpha_{i,t}=\sum_i\delta_{i,t}=0$. En los modelos dinámicos esta restricción se impone en cada paso temporal ([footBayes](https://search.r-project.org/CRAN/refmans/footBayes/html/stan_foot.html); [Macrì-Demartino et al. 2025](https://arxiv.org/pdf/2508.05891)).

### Dixon–Coles: una perturbación local que conserva masa

Dixon y Coles (1997) multiplican la pmf independiente por un factor que solo actúa en las cuatro celdas $\{0,1\}^2$ ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)):

$$
\tau_{\lambda,\mu}(x,y)=\begin{cases}1-\lambda\mu\rho & (0,0)\\ 1+\lambda\rho & (0,1)\\ 1+\mu\rho & (1,0)\\ 1-\rho & (1,1)\\ 1 & \text{en otro caso,}\end{cases}
\qquad \max\!\left(-\tfrac1\lambda,-\tfrac1\mu\right)\le\rho\le\min\!\left(\tfrac{1}{\lambda\mu},1\right).
$$

El rango admisible de $\rho$ depende del partido. Hay una verificación algebraica elegante para el artículo: como $p(0,1)=\mu\,p(0,0)$, $p(1,0)=\lambda\,p(0,0)$ y $p(1,1)=\lambda\mu\,p(0,0)$ bajo independencia, las correcciones suman $p(0,0)\,\rho\,[-\lambda\mu+\lambda\mu+\lambda\mu-\lambda\mu]=0$. Por lo tanto **$\tau$ conserva la masa exactamente**, aunque las marginales dejan de ser Poisson. En la práctica $\rho$ sale levemente negativo, del orden de **$-0{,}13$** en un ejemplo de la EPL, lo que infla el 0-0 y el 1-1 ([goalmodel](https://github.com/opisthokonta/goalmodel)). La versión original también pondera los partidos antiguos dentro de una pseudo-verosimilitud,

$$
\ell_t(\theta)=\sum_{k:\,t_k<t} e^{-\xi(t-t_k)}\log\!\big[\tau_{\lambda_k,\mu_k}(x_k,y_k)\,\mathrm{Pois}(x_k;\lambda_k)\,\mathrm{Pois}(y_k;\mu_k)\big],
$$

donde $\xi$ no se estima por máxima verosimilitud. Se elige maximizando el log-score predictivo 1X2 fuera de muestra ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)). Los valores útiles están entre **0,001 y 0,003 por día**; con el valor de ejemplo, 0,0019, la vida media es $\ln 2/0{,}0019\approx 365$ días ([goalmodel](https://github.com/opisthokonta/goalmodel)). Conviene leer $\xi$ como el *ancho de banda* de una verosimilitud local en el tiempo. Su contraparte generativa es el paseo aleatorio de la sección siguiente, y esa equivalencia vale una proposición en el artículo. La crítica central a Dixon–Coles es estructural: como solo toca celdas con a lo sumo dos goles en total, **las probabilidades de Over/Under 2,5 no dependen de $\rho$**. El modelo Mar-Co de Petretta et al., que condiciona un marcador en el otro mediante $\psi_Y(\mu,h)=\exp[\theta_1+\theta_2\log\mu+\theta_3\,\mathrm{logit}F_\mu(h)]$, supera claramente a Dixon–Coles en U/O 2,5 y empata en 1X2 ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)).

### Poisson bivariada, copulas y el signo de la dependencia

La Poisson bivariada (Karlis & Ntzoufras 2003) se construye por reducción trivariada: $X=W_1+W_3$, $Y=W_2+W_3$ con $W_\ell\sim\mathrm{Pois}(\lambda_\ell)$ independientes. Su pmf es

$$
P(x,y)=e^{-(\lambda_1+\lambda_2+\lambda_3)}\frac{\lambda_1^x\lambda_2^y}{x!\,y!}\sum_{k=0}^{\min(x,y)}\binom{x}{k}\binom{y}{k}k!\left(\frac{\lambda_3}{\lambda_1\lambda_2}\right)^k,
$$

con $\mathrm{Cov}(X,Y)=\lambda_3\ge 0$ ([Macrì-Demartino et al. 2025](https://arxiv.org/pdf/2508.05891)). La construcción **solo admite correlación no negativa**. Sin embargo, en 9.130 partidos de las cinco grandes ligas (2014/15–2018/19) la correlación de Pearson entre goles locales y visitantes es **$-0{,}085$** ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)). Esa cifra marginal mezcla heterogeneidad de fuerzas, así que la contradicción es solo aparente. Aun así, cabe esperar un $\hat\lambda_3$ cercano a cero. Koopman y Lit estiman $\gamma=\lambda_3\approx 0{,}097$ en la EPL dentro de un modelo dinámico ([Koopman & Lit 2015](https://papers.tinbergen.nl/12099.pdf)). La reducción trivariada sugiere un algoritmo EM natural con $W_3$ latente: el paso E es $s_k=\lambda_{3k}\,\mathrm{BP}(x_k-1,y_k-1)/\mathrm{BP}(x_k,y_k)$ y el paso M ajusta dos GLM Poisson sobre $x_k-s_k$ e $y_k-s_k$. Es una derivación propia que falta contrastar con [Karlis & Ntzoufras 2005 (JSS)](https://www.jstatsoft.org/index.php/jss/article/view/v014i10/v14i10.pdf).

La variante con diagonal inflada (DIBP) mezcla $(1-\omega)\,\mathrm{BP}+\omega\,\mathbb 1\{x=y\}D(x)$. El parámetro $\omega$ mide directamente la masa de "empate excedente" ([Macrì-Demartino et al. 2025](https://arxiv.org/pdf/2508.05891)). La Skellam modela la diferencia $Z=X-Y$, con $P(Z=z)=e^{-(\lambda_1+\lambda_2)}(\lambda_1/\lambda_2)^{z/2}I_{|z|}(2\sqrt{\lambda_1\lambda_2})$, y **elimina el choque común $W_3$**. Pero pierde los goles a favor, y varios reglamentos los usan para desempatar. Por eso el simulador necesita un modelo del marcador completo.

Las copulas separan las marginales de la dependencia y admiten signo negativo: McHale y Scarf (2011) encuentran dependencia negativa en partidos internacionales ([Petretta et al. 2022](https://arxiv.org/pdf/2103.07272)). Con marginales discretas, la pmf conjunta sale de la fórmula del rectángulo,

$$
P(x,y)=C_\theta(F_X(x),F_Y(y))-C_\theta(F_X(x^-),F_Y(y))-C_\theta(F_X(x),F_Y(y^-))+C_\theta(F_X(x^-),F_Y(y^-)),
$$

típicamente con la copula de Frank. El punto teórico fino es que, por el teorema de Sklar, la copula solo es única sobre $\mathrm{Ran}F_X\times\mathrm{Ran}F_Y$, una grilla numerable. Por eso $\theta$ pierde la interpretación habitual de "fuerza de dependencia" y los estimadores basados en rangos dejan de ser válidos. Esta afirmación se apoya en Genest y Nešlehová (2007), que no se verificó en esta investigación. Dixon–Coles con $\rho<0$ y una copula con $\theta<0$ no se contradicen: la primera es una asociación positiva *local* en la diagonal baja, la segunda una dependencia negativa global.

### Dispersión, empates, xG y localía

Condicional a las fuerzas, los goles parecen levemente **sub**-dispersos. Eso descarta la binomial negativa, que solo representa sobre-dispersión. Favorece en cambio a la Conway–Maxwell–Poisson, $P(n)=\lambda^n/[(n!)^\upsilon Z(\lambda,\upsilon)]$, donde $\upsilon>1$ da sub-dispersión y el parámetro $\lambda$ no es la media ([goalmodel](https://github.com/opisthokonta/goalmodel)). También favorece al conteo Weibull, que surge de un proceso de renovación con tiempos entre goles Weibull. En goles visitantes de la EPL 2009/10–2016/17, el Weibull mejora el log-likelihood de $-4364{,}3$ a $-4350{,}1$ ($\chi^2_1=28{,}48$, $p=9{,}4\times10^{-8}$) y no se rechaza por bondad de ajuste ($p=0{,}21$), mientras que la Poisson se rechaza con fuerza ([Countr](https://mirror.linux.duke.edu/cran/web/packages/Countr/vignettes/exampleFootball.pdf)). Ese ajuste es *incondicional*, y la heterogeneidad entre partidos infla la varianza marginal. La afirmación sobre sub-dispersión condicional necesita una fuente directa. La teoría de renovación da el puente matemático: el índice de dispersión tiende a $(\sigma_T/\mu_T)^2$, de modo que un hazard creciente ($c>1$) produce sub-dispersión.

El xG no tiene, en las fuentes revisadas, una estimación cuantitativa revisada por pares de su ganancia en RPS sobre los goles. Mead et al. (2023) lo describen solo como "superior predictor" ([STORRE](https://dspace.stir.ac.uk/handle/1893/35242)). El diseño coherente lo trata como **medición ruidosa de la misma tasa latente**: $x_k\sim\mathrm{Pois}(\lambda_k)$ y $\mathrm{xG}_k\sim\mathrm{Gamma}(\kappa\lambda_k,\kappa)$, de modo que $E[\mathrm{xG}_k]=\lambda_k$ y la precisión $\kappa$, estimada, cuantifica cuánto informa el xG.

La localía debe ser variable en el tiempo. De 26 estudios sobre partidos a puerta cerrada durante el COVID, **18 reportan una caída de la ventaja local y ninguno un aumento** ([Leitner et al. 2023](https://link.springer.com/10.1007/s11301-021-00254-5)). Una especificación jerárquica $\eta_i=\eta_0+\epsilon_i$, con $\epsilon_i\sim\mathcal N(0,\sigma_\eta^2)$, más una covariable de público, identificable gracias a ese experimento natural, cubre ambos hechos.

### El modelo de goles que anida a la literatura

Todo lo anterior cabe en una sola verosimilitud:

$$
P(x,y)=(1-\omega)\,\frac{\tau_\rho(x,y)\,f_\upsilon(x;\lambda_k)\,f_\upsilon(y;\mu_k)}{Z_k}+\omega\,\mathbb 1\{x=y\}\,D(x),
$$

donde $f_\upsilon$ es CMP y $Z_k$ renormaliza, porque $\tau$ solo conserva la masa con marginales Poisson. Maher corresponde a $\omega=\rho=0$, $\upsilon=1$ con fuerzas estáticas. Dixon–Coles agrega $\rho$ y $\xi$. Al reemplazar $\tau$ por $\lambda_3$ se obtiene Karlis–Ntzoufras, y con $\omega>0$, DIBP. Ninguna fuente establece la identificabilidad conjunta de $(\rho,\omega,\upsilon)$; los tres actúan sobre celdas bajas y empates. Lo prudente es elegir una sola corrección de dependencia mediante LFO-CV. En la temporada 2024/25, los modelos dinámicos ponderados BP y DIBP lograron el menor RPS en la mayoría de las ventanas (por ejemplo, **0,205** en la segunda mitad de la EPL), aunque las propias tablas del preprint muestran excepciones ([Macrì-Demartino et al. 2025](https://arxiv.org/pdf/2508.05891)).

## La fuerza es un proceso estocástico y Elo es un filtro aproximado

### Dinámica dentro y entre temporadas

Koopman y Lit (2015) modelan ataque y defensa como AR(1) semanales con observación Poisson bivariada. En la EPL 2003/04–2009/10 estiman **$\phi_\alpha\approx0{,}9985$ y $\phi_\beta\approx0{,}9992$** y una localía $\delta\approx0{,}364$ ([Koopman & Lit 2015](https://papers.tinbergen.nl/12099.pdf)). Dentro de la temporada eso es prácticamente un paseo aleatorio, pero implica una persistencia de $0{,}9985^{38}\approx0{,}94$ por temporada. La lectura en tiempo continuo pasa por el proceso de Ornstein–Uhlenbeck $d\theta=-\kappa(\theta-m)\,dt+\sigma\,dW$, que se discretiza exactamente como

$$
\theta_{t+\Delta}=m+e^{-\kappa\Delta}(\theta_t-m)+\varepsilon,\qquad \varepsilon\sim\mathcal N\!\Big(0,\tfrac{\sigma^2}{2\kappa}\big(1-e^{-2\kappa\Delta}\big)\Big).
$$

Con $\phi=0{,}9985$ semanal se obtiene $\kappa\approx0{,}0015$ por semana, es decir, una vida media de reversión de unas **460 semanas**. La reversión es despreciable dentro de una temporada y relevante solo a lo largo de varias. Por eso conviene seguir a Glickman y Stern (1998): **paseo aleatorio en días calendario dentro de la temporada y una transición explícita en el receso** ([Glickman & Stern 1998](https://doi.org/10.1080/01621459.1998.10474084)),

$$
\theta_{i,s,w+1}=\theta_{i,s,w}+\varepsilon_{i,w},\qquad
\theta_{i,s+1,1}=m_{\ell}+\varrho\,(\theta_{i,s,\mathrm{fin}}-m_\ell)+\varepsilon'_{i,s}.
$$

En esta transición, $\varrho$ y la varianza de $\varepsilon'$ se estiman jerárquicamente entre ligas. Koopman y Lit tratan como faltantes las temporadas en que un equipo no está en la liga. Ningún artículo central propone una inicialización principista para los ascendidos ([Koopman & Lit 2015](https://papers.tinbergen.nl/12099.pdf)). La propuesta aquí es un prior $\theta_{i,s,1}\sim\mathcal N(a_0+a_1\log\mathrm{MV}_{i,s}+\text{offset}_{\text{ascenso}},\tau^2)$, construido con valores de mercado, que según Peeters (2018) predicen mejor que Elo y el ranking FIFA en selecciones ([Peeters 2018](https://ideas.repec.org/a/eee/intfor/v34y2018i1p17-29.html)). Para respetar la suma cero en un modelo dinámico basta proyectar las innovaciones al subespacio correspondiente, $w_t\sim\mathcal N\big(0,\sigma^2(I-\tfrac1n\mathbf 1\mathbf 1^\top)\big)$, una gaussiana degenerada de rango $n-1$.

### Elo, Glicko, TrueSkill y el filtro de Kalman extendido

Para el artículo, la unificación más bonita es que **todo sistema de rating equivale a una verosimilitud, un prior gaussiano de paseo aleatorio y una regla de proyección gaussiana**. Elo es un paso de gradiente estocástico sobre la log-verosimilitud de Bradley–Terry: con $z=\theta_i-\theta_j+h$ y $P(y=1)=\sigma(z)$, se tiene $\partial\ell/\partial\theta_i=y-\sigma(z)$, y el paso $\theta_i\leftarrow\theta_i+K(y-E)$ *es* Elo. Contar el empate como ½ equivale al modelo de Davidson con $\kappa=2$, que fija en torno a 50% la frecuencia de empates entre equipos parejos. Szczecinski y Djebbi lo califican de irreal y proponen $\kappa\approx2p_D/(1-p_D)$ ([Szczecinski & Djebbi 2020](https://arxiv.org/abs/1910.06081)). Un trabajo de abril de 2026 muestra que las visiones "regla de retroalimentación" y "ML online" coinciden exactamente solo para resultados binarios, y recomienda desacoplar el modelo de ranking del de predicción ([Szczecinski 2026](https://arxiv.org/abs/2604.03840)).

Glicko infla la varianza antes de cada periodo ($\sigma_t^2=\sigma^2+c^2t$), que es el paso de predicción de Kalman. Luego actualiza $\sigma'^2=(1/\sigma^2+1/d^2)^{-1}$ y $\mu'=\mu+q\sigma'^2\sum_j g(\sigma_j^2)(s_j-E_j)$, con $g(\sigma^2)=1/\sqrt{1+3q^2\sigma^2/\pi^2}$. Es un filtro de campo medio que ignora las correlaciones posteriores ([Glickman 1999](http://www.glicko.net/research/glicko.pdf)). Linealizando la observación Bernoulli en la media previa se obtiene un EKF escalar:

$$
\mu_i\leftarrow\mu_i+\frac{v_i}{1+\varsigma(v_i+v_j)}\,(y-E),\qquad v_i\leftarrow v_i-\frac{\varsigma\, v_i^2}{1+\varsigma(v_i+v_j)},\qquad \varsigma=E(1-E).
$$

En estado estacionario esto es Elo con $K=v^*/(1+2\varsigma v^*)$. Es la estructura del *Simplified Kalman Filter* de Szczecinski y Tihon ([SKF, JQAS 2023](https://ideas.repec.org/a/bpj/jqsprt/v19y2023i4p295-315n5.html)). La derivación es propia y es consistente con las ecuaciones de Glickman. Aplicada a goles Poisson con $\log\lambda=\alpha_i-\delta_j+h$, la *score* es $y-\lambda$ y la información de Fisher es $\lambda$. Resulta un **"Elo de ataque/defensa"**, $\alpha_i\leftarrow\alpha_i+\frac{v_\alpha}{1+\lambda(v_\alpha+v_\delta)}(\text{goles}-\lambda)$, que sirve como filtro rápido y aproxima el modelo de Koopman–Lit. TrueSkill, en cambio, iguala momentos de la posterior probit exacta (EP/ADF) ([Herbrich et al. 2007](https://papers.nips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)). Los pi-ratings (Constantinou & Fenton 2013) se leen como gradiente sobre una pérdida robusta tipo log y no sobre una verosimilitud ([DOI](https://doi.org/10.1515/jqas-2012-0036)). Las constantes $\lambda\approx0{,}035$, $\gamma\approx0{,}7$, $c=3$ quedan por verificar. Finalmente, el modelo GAS de Koopman y Lit (2019), $f_{t+1}=\omega+Bf_t+As_t$, con $B=I$ y escalamiento identidad *es literalmente* Elo. Eso conecta los modelos dirigidos por observaciones con los dirigidos por parámetros ([TI 17-062](https://papers.tinbergen.nl/17062.pdf)).

### Cambios de régimen y priors gaussianos sobre el tiempo

Los despidos de entrenador y los mercados de pases sugieren saltos. La detección bayesiana online de cambios (Adams & MacKay 2007) da una recursión exacta sobre la longitud de racha $r_t$:

$$
P(r_t,x_{1:t})=\sum_{r_{t-1}}P(r_t\mid r_{t-1})\,\pi_t^{(r_{t-1})}\,P(r_{t-1},x_{1:t-1}),\qquad P(r_t=0\mid r_{t-1})=H(r_{t-1}),
$$

y eso permite usar un hazard *informativo* que se eleva la semana de un cambio de DT ([arXiv:0710.3742](https://arxiv.org/abs/0710.3742)). Con verosimilitud Poisson aproximada por una gaussiana, el resultado es un banco de filtros de Kalman indexado por la racha, o un filtro de partículas Rao–Blackwellizado. Las alternativas continuas compatibles con NUTS son tres: innovaciones Student-t escritas como mezcla de escalas, $\eta_t=\varepsilon_t/\sqrt{\lambda_t}$ con $\lambda_t\sim\mathrm{Gamma}(\nu/2,\nu/2)$; volatilidad estocástica sobre el incremento; y priors horseshoe sobre los incrementos, el análogo bayesiano del *trend filtering*. **No se encontró ninguna aplicación revisada por pares de BOCPD a la fuerza de equipos de fútbol.** La única aplicación deportiva documentada es una tesis de Leiden sin fútbol ([Haitink 2022](https://theses.liacs.nl/pdf/2021-2022-HaitinkM.pdf)), así que aquí hay un hueco real. Por otro lado, un GP temporal con kernel Matérn de orden $p+\tfrac12$ tiene representación exacta en espacio de estados: OU es Matérn-½ y Matérn-3/2 es un estado bidimensional, con costo $O(T)$ vía Kalman ([Hartikainen & Särkkä 2010](https://doi.org/10.1109/MLSP.2010.5589113)). Tampoco se encontró un artículo de fútbol que use priors GP sobre el tiempo.

### Inferencia: qué motor para qué tarea

Con 7 ligas, unos 20 equipos, 2 estados por equipo, 38 semanas y 5 temporadas, el modelo completo tiene **del orden de 53.000 latentes**. Un paseo aleatorio no centrado con NUTS es factible en minutos, aunque esto es una estimación y no una medición. Para actualizaciones intra-día basta el EKF/SKF. Si se quiere tratar de forma genuinamente online los hiperparámetros estáticos, las opciones son IBIS (Chopin 2002) o SMC² (Chopin, Jacob & Papaspiliopoulos 2013). IBIS resulta muy barato si los estados se marginalizan con la aproximación de Laplace, que Stan incorporó de forma embebida en **CmdStan 2.39 (19 de mayo de 2026)** ([Stan blog](https://blog.mc-stan.org/); [Chopin 2002](https://doi.org/10.1093/biomet/89.3.539); [SMC²](https://doi.org/10.1111/j.1467-9868.2012.01046.x)). Una advertencia directamente relevante: **la VI de campo medio subestima la correlación entre estados consecutivos del paseo aleatorio, y con ello la incertidumbre de la tabla final.** Si se usa VI, debe tener guías de rango completo o markovianas.

## Estilos y ligas viven en el mismo grafo de comparaciones

### Del "head-to-head" al término bilineal

No existe evidencia revisada por pares de que el historial directo prediga algo una vez modeladas fuerza y localía. Lo que circula en la web sobre H2H es contenido de blogs de apuestas ([sportbotai](https://www.sportbotai.com/blog/head-to-head-records-soccer-prediction-models-ai-betting-value)). Un efecto diádico libre $\gamma_{ij}$ se estimaría con un partido por temporada en cada dirección, lo que no tiene sentido. La formulación rigurosa es un **término bilineal dirigido** de bajo rango, análogo al término multiplicativo de los modelos AMEN de Hoff:

$$
\log\lambda^H_{ij}=\mu_0+\eta+\alpha_i-\delta_j+u_i^\top v_j,\qquad \log\lambda^A_{ij}=\mu_0+\alpha_j-\delta_i+u_j^\top v_i ,
$$

donde $u_i$ es el "estilo de ataque" y $v_j$ la "susceptibilidad defensiva". Lo matemáticamente interesante es que **el logit de victoria inducido contiene la parte antisimétrica $u_i^\top v_j-u_j^\top v_i$, que es exactamente el modelo *blade-chest* de Chen y Joachims**, con *blade* $=u$ y *chest* $=v$ ([Duan et al.](https://arxiv.org/pdf/2409.19325)). En forma matricial, $AB^\top-BA^\top$ es antisimétrica de rango $2K$ ([Lee & Chen, JMLR 2026](https://www.jmlr.org/papers/volume27/25-0217/25-0217.pdf)). Una variante con embedding compartido, $u_i^\top v_j=z_i^\top\Sigma z_j$, $\Sigma=\begin{pmatrix}0&s\\-s&0\end{pmatrix}$, da con $K=2$ un **producto cruz 2-D** $s\,(z_i\times z_j)$: una geometría de "piedra-papel-tijera" entre estilos que se puede visualizar.

La identificabilidad requiere tres arreglos. El término tiene una simetría de gauge $\mathrm{GL}(K)$, $u\mapsto A^\top u$, $v\mapsto A^{-1}v$, así que solo $\Gamma=UV^\top$ está identificado; en MCMC conviene postprocesar cada draw por SVD con alineamiento de Procrustes. Además, $\Gamma$ se confunde con los efectos principales salvo que esté **doblemente centrada** ($\sum_iu_i=\sum_jv_j=0$), como en la descomposición ANOVA de dos vías. Por último, hay que regularizar con priors ARD o horseshoe sobre los valores singulares, calibrados para que la desviación estándar a priori de $u_i^\top v_j$ quede en torno a 0,05–0,15 en log-tasa, es decir, ±5–15% en goles esperados. Ese rango es una sugerencia que debe comprobarse con simulación predictiva a priori.

Las expectativas deben ser modestas. En StarCraft II el modelo antisimétrico de bajo rango sube la precisión de **0,713 a 0,766**, con un 70% de tripletas intransitivas; en el tenis ATP, Bradley–Terry *gana* (0,658 frente a 0,652) ([Lee & Chen 2026](https://www.jmlr.org/papers/volume27/25-0217/25-0217.pdf)). El fútbol, ruidoso y mayormente transitivo, probablemente se parece más al tenis. El término de estilos debe presentarse entonces como **hipótesis a contrastar**, con LFO-CV frente al modelo sin interacción y con un nulo por permutación de pares dentro de cada temporada. Los estilos pueden anclarse parcialmente a descriptores observados mediante $u_i\sim\mathcal N(Bx_i,\tau^2I)$, con métricas de posesión como las ocho de Chacoma et al. ([arXiv:2606.21814](https://arxiv.org/html/2606.21814)). **No se encontró ningún modelo de fútbol publicado que ajuste $u_i^\top v_j$ y reporte resultados fuera de muestra.**

### Hodge: gradiente, rizo y la componente armónica entre ligas

HodgeRank descompone un flujo de aristas $Y_{ij}=-Y_{ji}$ (por ejemplo, diferencia de goles ajustada por localía) en tres componentes ortogonales: $Y=\mathrm{grad}\,r^*\oplus\mathrm{curl}^*\Phi\oplus h$ ([Jiang, Lim, Yao & Ye 2011](https://www.stat.uchicago.edu/~lekheng/work/mathprog.pdf)). En una liga completa ($K_{20}$) la componente armónica se anula. Chacoma et al. aplican esto a métricas de estilo de las cinco grandes ligas 2017/18 y encuentran **inconsistencia $I>0{,}6$ en Alemania** para contraataque y tiros. También encuentran que la inconsistencia se relaciona negativamente con lo bien que el gradiente recupera la tabla (pendiente de Kendall $-0{,}32$) ([arXiv:2606.21814](https://arxiv.org/html/2606.21814)). La contribución nueva que se propone aquí tiene dos piezas. La primera es aplicar Hodge a flujos de *goles o xG* sobre el grafo multi-liga con copas continentales, donde los ciclos largos con pocos triángulos hacen **no trivial la componente armónica** ($\dim\ker\Delta_1$ es el primer número de Betti del complejo de cliques), que mide la inconsistencia global de las comparaciones entre ligas. La segunda es calibrar la razón de ciclicidad $C=1-\|\mathrm{grad}\,r^*\|_w^2/\|Y\|_w^2$ contra su distribución bajo el modelo transitivo ajustado mediante bootstrap paramétrico, porque **el ruido Poisson por sí solo genera rizo**.

### La comparación entre ligas es un problema de redes eléctricas

El modelo de Massey $y_m=r_i-r_j+hs_m+\varepsilon_m$ tiene ecuaciones normales $Lr=b$, donde $L=D-A$ es el laplaciano del multigrafo de partidos. De ahí sale una teoría completa de identificabilidad e incertidumbre, que se deriva así:

$$
\ker L=\mathrm{span}\{\mathbf 1_C\},\qquad \mathrm{Var}(\hat r_i-\hat r_j)=\sigma^2(e_i-e_j)^\top L^\dagger(e_i-e_j)=\sigma^2\,R_{\mathrm{eff}}(i,j),
$$

$$
\mathrm{Kf}=\sum_{i<j}R_{\mathrm{eff}}(i,j)=n\sum_{k\ge2}\frac1{\lambda_k},\qquad R_{\mathrm{eff}}(i,j)\le\frac{2}{\lambda_2}.
$$

La primera identidad dice que los ratings están identificados solo dentro de cada componente conexa. Si ningún club chileno jugó un partido CONMEBOL en la ventana, el nivel de la liga chilena relativo a Europa **no está identificado por resultados**. La segunda dice que la varianza de una diferencia de ratings es la resistencia efectiva entre los dos nodos, con cada partido como un conductor unitario. Hendrickx, Olshevsky y Saligrama prueban que el error de estimación en BTL está gobernado por la **resistencia efectiva y no por la brecha espectral**, con cotas superior e inferior que coinciden. El caso extremo de su comparación es el grafo *barbell* (dos clusters densos unidos por un puente delgado), en el que las cotas espectrales son mucho peores: es exactamente la topología Europa–Sudamérica ([Hendrickx et al., ICML 2019](https://arxiv.org/abs/1902.00141)). Con verosimilitud Poisson, la información de Fisher es un laplaciano ponderado con conductancias $\lambda_m$, de modo que la lógica se mantiene localmente.

El puente intercontinental es delgado y sorprendente. En el Mundial de Clubes 2025, **Botafogo venció 1-0 al PSG y Flamengo 3-1 al Chelsea** ([Korea JoongAng Daily](https://www.koreajoongangdaily.com/sports/brazilian-argentinian-clubs-beat-predictions-to-dominate-club-world-cup/12284403)). Tras la primera jornada, los clubes CONMEBOL estaban 3V–3E–0D ([Flashscore](https://www.flashscore.com/news/soccer-fifa-club-world-cup-european-and-south-american-clubs-lead-the-way-as-global-gap-widens-at-club-world-cup/WCA0J4J8)). Son del orden de una decena de aristas cada cuatro años frente a decenas de miles de partidos domésticos. Una estimación gruesa, propia y por verificar con el fixture real, sitúa el error estándar de la diferencia "club de la Premier vs club chileno" cerca de **0,4 goles por partido** solo por el puente. Por eso los priors adicionales no son opcionales.

La arquitectura que sigue de esto descompone la fuerza jerárquicamente en un efecto de liga y una desviación del equipo, $\theta_i=\lambda_{\ell(i)}+\delta_i$ con $\lambda_\ell\sim\mathcal N(\kappa_{c(\ell)},\sigma_\lambda^2)$. Opta Power Rankings hace algo parecido con un Elo jerárquico donde el rating es la suma de equipo, liga, país y continente, y los niveles superiores solo se actualizan con partidos que cruzan grupos ([Opta Analyst](https://theanalyst.com/eu/2023/01/power-rankings-your-club-ranked/)). Las ligas quedan identificadas por tres tipos de aristas, todas como **conductancias en paralelo dentro del mismo GMRF**. El primero son los **partidos internacionales**. El segundo son las **transferencias como "aristas blandas"**: para cada jugador que pasa de la liga $\ell$ a la $\ell'$, $\Delta_p=\kappa(s_\ell-s_{\ell'})+\gamma^\top z_p+\eta_p$, siguiendo el diseño de incidencia ±1 de ElHabr, que estima Premier 0,975, La Liga 0,869 y Brasil 0,724 en escala z ([Tony ElHabr](https://tonyelhabr.rbind.io/posts/soccer-league-strength/)). El tercero son los **valores de mercado como "aristas a tierra"**, que eliminan el núcleo de $L$ y acotan toda $R_{\mathrm{eff}}$. Por monotonía de Rayleigh, **cada arista agregada reduce estrictamente todas las resistencias efectivas**. El vector de Fiedler del grafo global debería recuperar la partición en continentes y ligas, lo que daría una figura convincente para el artículo. No se encontró ningún trabajo que combine aristas de partidos y de transferencias en un solo modelo laplaciano. Es probablemente la contribución más original del proyecto.

## La matriz de posiciones exige reglas reales e incertidumbre paramétrica

### Tres familias de desempate y un comparador recursivo

| Liga (2025-26 / 2026) | Orden de desempate | Ganchos de play-off |
|---|---|---|
| Premier League (20) | Pts, DG, GF; luego H2H pts y goles de visita en H2H si afecta título/descenso/Europa | Play-off en cancha neutral (nunca usado) ([premierleague.com](https://www.premierleague.com/news/4183163)) |
| La Liga (20) | Pts, H2H pts, H2H DG, DG, GF, fair play ([Wikipedia](https://en.wikipedia.org/wiki/2025–26_La_Liga)) | — |
| Serie A (20) | Classifica avulsa: H2H pts, H2H DG, DG, GF, sorteo | Título con 2 empatados: partido único en casa del mejor clasificado según la avulsa, sin alargue. Puesto 17/18: ida y vuelta ([FIGC CU 331/A](https://figc.it/media/274515/331-deroga-art-51-noif-determinazione-classifica-campionato-serie-a-ss-2025-2026.pdf)) |
| Bundesliga (18) | Pts, DG, GF, H2H, H2H visita, goles de visita | 16.º contra 3.º de 2. Bundesliga, ida y vuelta con alargue ([Wikipedia](https://en.wikipedia.org/wiki/2025–26_Bundesliga)) |
| Ligue 1 (18) | Pts, DG, H2H pts, H2H DG, H2H GF, H2H visita, GF, visita, fair play | Barrage 16.º contra Ligue 2 ([Wikipedia](https://en.wikipedia.org/wiki/2025–26_Ligue_1)) |
| Chile (16, 30 fechas) | Pts, [play-off solo por el título], DG, victorias, GF, GF visita, rojas, amarillas, sorteo | Descienden 2; el 3.º juega contra el campeón de Copa Chile ([Wikipedia](https://en.wikipedia.org/wiki/2026_Liga_de_Primera); [T13](https://www.t13.cl/noticia/deportes13/los-ascensos-descensos-futbol-chileno-2026)) |
| Brasileirão (20) | **Victorias**, DG, GF, H2H, rojas, amarillas, sorteo ([Trivela, reglamento 2025](https://trivela.com.br/brasil/brasileirao-criterios-desempate/)) | Descienden 4; Libertadores para el top 5 más campeón y subcampeón de Copa do Brasil ([Wikipedia](https://en.wikipedia.org/wiki/2026_Campeonato_Brasileiro_Série_A)) |

La implementación correcta es un **comparador lexicográfico recursivo**: se particiona por puntos; en cada bloque empatado se calcula el siguiente criterio, restringiendo a los partidos internos del bloque si el criterio es H2H; luego se vuelve a particionar y se recurre. Los play-offs se simulan como partidos. El de Serie A por el título da $P(\text{gana})=P(H)+P(D)\,P(\text{penales})$. En Brasil el criterio de victorias vuelve importante modelar bien la tasa de empates, porque victorias y empates están negativamente correlacionados. Argentina, que se deja para después, requiere un simulador de dos niveles: zonas de 15, una llave eliminatoria con alargue y penales, la Tabla Anual, promedios con historia previa, partidos de desempate por descenso y la asignación en cascada de cupos continentales ([El Gráfico](https://www.elgrafico.com.ar/articulo/primera-division/99695/guia-para-entender-el-reglamento-de-la-temporada-2026)).

### Predictiva posterior, no plug-in: la ley de varianza total

El "supercomputador" de Opta simula **10.000 veces** los partidos restantes a partir de probabilidades fijas construidas con cuotas de apuestas y Power Rankings ([The Analyst](https://theanalyst.com/articles/opta-supercomputer-premier-league-2023-24-predictions/)). La versión bayesiana correcta es

$$
p_{ik}=P(\mathrm{pos}_i=k\mid\mathcal D)=\int P(\mathrm{pos}_i=k\mid\theta)\,\pi(\theta\mid\mathcal D)\,d\theta ,
$$

**muestreando $\theta$ una vez por temporada simulada y no una vez por partido**. Re-muestrear por partido equivale a plug-in en las probabilidades predictivas y destruye la correlación positiva entre los resultados de un mismo equipo. La justificación es la ley de varianza total para los puntos finales $\Pi_i$:

$$
\mathrm{Var}(\Pi_i)=\underbrace{E_\theta[\mathrm{Var}(\Pi_i\mid\theta)]}_{\propto M_i}+\underbrace{\mathrm{Var}_\theta(E[\Pi_i\mid\theta])}_{\approx M_i^2\,\mathrm{Var}_\theta(\mu_i)} .
$$

El segundo término crece **cuadráticamente** con los $M_i$ partidos restantes. El plug-in lo omite, de modo que sobreestima las probabilidades de título de los líderes al inicio de la temporada y subestima, a veces en órdenes de magnitud, colas como "equipo grande desciende". El paseo aleatorio debe propagarse también *hacia adelante*: $\theta_{T+h}\mid\theta_T$ en cada fecha futura. Es la versión principista de las simulaciones "calientes". No se encontró ningún estudio publicado que cuantifique esta sobreconfianza en matrices de posición de fútbol, así que demostrarla empíricamente es otro aporte posible.

### Cuántas simulaciones y cómo reducir varianza

Para una probabilidad, $\mathrm{SE}=\sqrt{p(1-p)/N}$: con $N=10^4$ el peor caso es 0,5 pp, y mostrar porcentajes enteros correctos exige $N\gtrsim10^5$. Una cota de Bonferroni sobre las 400 celdas de una matriz 20×20 lleva $z$ de 1,96 a 3,66, lo que multiplica $N$ por unos 3,5. Para eventos raros el error relativo es $\sqrt{(1-p)/(Np)}$, de modo que $p=10^{-5}$ con 10% de error relativo exige $10^7$ temporadas. Si se reutilizan $S$ draws posteriores en $R$ temporadas cada uno,

$$
\mathrm{Var}(\hat p)=\frac{\mathrm{Var}_\theta(p(\theta))}{S}+\frac{E_\theta[p(\theta)(1-p(\theta))]}{SR},
$$

y $R\approx10$–$100$ es razonable cuando el MCMC es caro. Las técnicas de reducción de varianza son las de Glasserman, citadas desde el libro y no desde esta investigación. **Números aleatorios comunes** para los escenarios "qué pasa si", donde solo cambia un partido y la covarianza es enorme. **Variables de control** con $E[\Pi_i\mid\theta]=\Pi_i^0+\sum_m(3p^{\text{win}}_{im}+p^{\text{draw}}_{im})$, conocida exactamente. **Estratificación** sobre el resultado de un partido clave, que de paso entrega las tablas condicionales que necesita la interfaz. **Rao–Blackwellización** de la última fecha mediante enumeración exacta de $3^{10}$ resultados. Para "equipo grande desciende" conviene muestreo por importancia que *inclina la fuerza del equipo* (la densidad de varianza cero es la ley condicional al evento), con el método de entropía cruzada para elegir la inclinación o *splitting* multinivel sobre la distancia a la zona de descenso. El QMC aporta poco, porque los integrandos son indicadores discontinuos.

### Importancia, competitividad y "qué necesita mi equipo"

Hay tres familias de medidas de importancia: deterministas ("¿todavía puede importar?"), por diferencia probabilística, y por asociación o información. La de Goossens, Beliën y Spieksma es $I_m(X)=2n_m(X)/A_m(X)^2$, con PLE que calculan la mejor y peor posición alcanzable de cada equipo. En Bélgica ningún formato de liga dominó en todos los criterios ([Goossens et al.](https://lirias.kuleuven.be/retrieve/45090)). Lahvička mide la asociación $\tau_c$ entre resultado y desenlace en simulaciones de la Premier 2000/01–2011/12. Encuentra que **un 42% de los partidos no tiene importancia alguna para el título** y que la media es de 0,0165 para el título y 0,0530 para el descenso ([Lahvička](https://mpra.ub.uni-muenchen.de/40998/1/MPRA_paper_40998.pdf)). La propuesta aquí es la versión informacional:

$$
I(R_m;Y)=H(Y)-\sum_r P(r)H(Y\mid R_m=r)=\sum_r P(r)\,\mathrm{KL}\big(P(Y\mid r)\,\|\,P(Y)\big)\le\log 3 .
$$

Es la "sorpresa" esperada de la tabla causada por el partido. La distancia $W_1$ entre las distribuciones de posición condicional y marginal se interpreta como "posiciones esperadas movidas". Se ha encontrado que los partidos más decisivos coinciden entre modelos, pero el resto de los partidos clave depende del modelo ([Corona et al. 2017](https://livrepository.liverpool.ac.uk/3006295/)).

Para la competitividad, Noll–Scully con 3-1-0 y probabilidad de empate $d$ tiene una desviación ideal por partido de $\sqrt{\mathrm{Var}[X]/G}$, con $E[X]=1{,}5-0{,}5d$ y $\mathrm{Var}[X]=4{,}5(1-d)+d-E[X]^2$. Con $d=0{,}27$ y $G=38$ eso da unos 0,21 puntos por partido (cálculo propio). A esto se suman HHI*, Gini y, sobre todo, la **entropía del campeón $H_t=-\sum_ip_{i,t}\log p_{i,t}$**, cuyo $e^{H_t}$ es un "número efectivo de candidatos" y cuya curva a lo largo de las fechas mide el suspenso de la temporada. Empíricamente, los índices de concentración ponderados predicen mejor la asistencia que los índices clásicos: la elasticidad de largo plazo de $SDC^I_K$ es de **−1,14**, mientras que NAMSI, HHI* y Gini resultan mayormente no significativos ([Manasis et al.](https://arxiv.org/abs/1507.00634)).

La pregunta "¿qué necesita mi equipo?" tiene un lado combinatorio con un resultado duro: **con 3-1-0, decidir la eliminación matemática es NP-completo, incluso con a lo sumo tres partidos restantes por equipo**, y polinomial con dos ([Bernholt et al.](https://image.informatik.htw-aalen.de/~thierauf/TI2/Netz/football.pdf); [Kern & Paulusma 2001](https://durham-repository.worktribe.com/output/1414988)). Con 2-1-0 el problema se reduce a flujo máximo, como la eliminación en béisbol. En la práctica, programas enteros con binarias $h_m+d_m+a_m=1$ e indicadores $y_i$ de "terminar sobre $t$" (minimizando $\sum_iy_i$ para la mejor posición) resuelven ligas reales rápido. Ribeiro y Urrutia lo aplicaron al Brasileirão 2002 ([Optimization Online](https://optimization-online.org/2003/10/754/)), y Goossens et al. manejan los desempates con el truco de contar 3,01 puntos por victoria ([Goossens et al.](https://lirias.kuleuven.be/retrieve/45090)). Del lado probabilístico, con $\Pi^+_t$ los puntos que sumará el equipo, se define la curva $f(x)=P(A\mid\Pi^+_t=x)$ y un **número mágico probabilístico $x^*_\alpha=\min\{x:f(x)\ge\alpha\}$**. A eso se agrega, para cada partido ajeno, $\Delta_{m,o}=P(A\mid o)-P(A)$, que dice "a quién hinchar".

## Dentro del partido: tarjetas rojas sí, contagio de goles casi no

El esqueleto en vivo es un proceso de conteo con intensidad dependiente del estado. Dixon y Robinson (1998) usaron un proceso de nacimiento bidimensional con intensidades que cambian según el marcador más una tendencia temporal ([Volf 2009](https://ro.utia.cas.cz/files/Soutez_09/Volf/VOLF_IMAMAN2009.pdf)). La tasa total crece de forma aproximadamente lineal, cae un 50% en los minutos 1 y 46, y tiene picos en los minutos 45 y 90 por el tiempo añadido ([Heuer & Rubner 2012](https://ar5iv.labs.arxiv.org/html/1207.4471)). Con intensidad determinista, los goles restantes siguen siendo Poisson con media igual a la integral de la intensidad, así que la probabilidad en vivo es cerrada:

$$
P(\text{local gana}\mid x,y,t)=\sum_{d>y-x}\mathrm{Sk}\big(d;\Lambda_H(t,T),\Lambda_A(t,T)\big),\qquad \Lambda_H(t,T)=\int_t^T\lambda_H(s)\,ds .
$$

Con log-intensidad lineal en regresores, $\ln\lambda(t)=\sum_i\xi_i\psi_i(t)$, la log-verosimilitud del proceso de conteo, $\sum_l\ln\lambda(t_l)-\int_0^T\lambda$, es **estrictamente cóncava** (Teorema 1 de [Maia et al. 2023](https://arxiv.org/pdf/2312.04338)). Esas mismas estimaciones dan efectos de marcador modestos: ×0,908 para el que gana por uno y ×1,101 para el que pierde por uno. Para la tarjeta roja dan **×1,44 en la tasa del rival y ×0,70 en la del equipo sancionado**, con rojas que llegan según $\lambda(t)=e^{\xi_0}t^{\xi_1}$, $\xi_1\approx1{,}43$ ([Maia et al.](https://arxiv.org/pdf/2312.04338)). En Mundiales, la roja del rival eleva la tasa en +124% y la propia la reduce en −47% ([Červený et al. 2017](https://repub.eur.nl/pub/109448/REPUB_109448_OA.pdf)). Las magnitudes varían por muestra y especificación, así que deben estimarse con datos propios. En cuanto el estado entra en la intensidad, la Skellam deja de valer, y lo elegante es resolver las **ecuaciones de Kolmogorov hacia adelante** sobre una retícula truncada:

$$
\partial_tp(x,y,t)=\lambda_{x-1,y}(t)\,p(x-1,y,t)+\mu_{x,y-1}(t)\,p(x,y-1,t)-\big[\lambda_{xy}(t)+\mu_{xy}(t)\big]p(x,y,t),
$$

ampliadas con las dimensiones $(r_H,r_A)$ para las rojas. Es un sistema triangular inferior que se resuelve en milisegundos. Su forma retrógrada es la EDP de valoración de Divos et al., donde el mercado Poisson en vivo resulta completo y libre de arbitraje: $\partial_tX=-\lambda_1\delta_1X-\lambda_2\delta_2X$, con deltas de cobertura $\varphi^i=\delta_iX$ ([Divos et al. 2018](https://arxiv.org/pdf/1811.03931)). La actualización intra-partido de la fuerza tiene una forma conjugada simple: con prior $\mathrm{Gamma}(a,b)$ sobre un multiplicador y $n$ goles observados hasta $t$, la posterior es $\mathrm{Gamma}(a+n,b+G(t))$, y los goles restantes pasan a ser binomial negativa.

La hipótesis Hawkes de "contagio" debe formularse para **contrastarse, no asumirse**:

$$
\lambda_k(t)=\mu_k(t)+\sum_{l\in\{H,A\}}\sum_{t^l_i<t}\alpha_{kl}\beta_{kl}e^{-\beta_{kl}(t-t^l_i)},\qquad \ell=\sum_k\Big[\sum_i\ln\lambda_k(t^k_i)-\int_0^T\lambda_k\Big].
$$

Aquí $\mu_k(t)$ es la línea base NHPP. La evidencia apunta contra $\alpha_{kk}>0$ en goles. En 43 temporadas de la Bundesliga, el equipo que anota no cambia su tasa y el que recibe baja cerca de un 10% durante unos 10 minutos. Además, el orden de los goles no importa: tras un 1-1 al descanso, la diferencia en el segundo tiempo es 0,43 ± 0,07 frente a 0,46 ± 0,07 ([Heuer & Rubner](https://ar5iv.labs.arxiv.org/html/1207.4471)). En más de 500.000 eventos de la EPL, **los tiempos no se agrupan (la K de Ripley queda bajo Poisson), pero las marcas sí se excitan fuertemente**: los tiros casi siempre los gatillan eventos previos ([Narayanan et al. 2023](https://arxiv.org/pdf/2103.04647)). Daniel y Ritov encuentran que el primer gol *frena* el segundo antes del minuto 52 y lo acelera después, sin distinguir quién anotó ([Daniel & Ritov 2013](https://ar5iv.arxiv.org/html/1207.6796)), lo que se explica mejor por la tasa creciente y los efectos de estado que por autoexcitación. La conclusión de diseño es que **Hawkes rinde en flujos de tiros y xG, no en goles**. Una prueba de "rachas" debe compararse contra un nulo Poisson heterogéneo y no homogéneo, porque tiempos entre goles exponenciales es justo lo que produce un proceso de Poisson. El mejor resultado empírico en vivo usa coeficientes con paseo aleatorio sobre 100 tramos del partido y baja el RPS de 0,1571 a **0,1338** con variables contextuales ([Robberechts et al. 2021](https://arxiv.org/pdf/1906.05029)).

## Evaluar con reglas propias, no con intuición

Una regla $S$ es propia si $E_{Y\sim Q}S(Q,Y)\ge E_{Y\sim Q}S(P,Y)$. Savage y Gneiting–Raftery la caracterizan mediante una $G$ convexa: $S(p,i)=G(p)-\langle\nabla G(p),p\rangle+\partial_iG(p)$. Con $G=\sum p_j^2-1$ se obtiene Brier y con $G=\sum p_j\log p_j$ el log-score ([Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437)). Hay tres hechos para el artículo. Primero, el RPS es un CRPS discreto normalizado,

$$
\mathrm{RPS}=\frac{1}{r-1}\sum_{k=1}^{r-1}(F_k-O_k)^2=\frac{\mathrm{CRPS}}{r-1},\qquad \mathrm{CRPS}(F,y)=E_F|X-y|-\tfrac12E_F|X-X'| ,
$$

es decir, una distancia de Cramér o energía. Segundo, **$W_1(F,\delta_y)=E_F|X-y|$ es impropia**: su esperanza se minimiza en una masa puntual en la mediana de $Q$, y el término $-\tfrac12E|X-X'|$ es justamente lo que la corrige. Tercero, el log-score es la única regla propia *local*. En las simulaciones de Wheatcroft identifica el sistema verdadero mejor que Brier y RPS, mientras que el RPS "aprovecha mal los datos adicionales" ([Wheatcroft 2019/2021](https://arxiv.org/abs/1908.08980)). De ahí la recomendación: **log-score principal, RPS para comparar con la literatura**. Para posiciones finales se usa el RPS de 20 clases, el log-score con piso de Monte Carlo, Brier sobre eventos binarios (título, top-4, descenso) y el score de energía sobre el vector de posiciones o puntos para evaluar la distribución conjunta.

La calibración se mide con diagramas **CORP** (recalibración isotónica PAV), que descomponen cualquier score propio como $\bar S=\mathrm{MCB}-\mathrm{DSC}+\mathrm{UNC}$ sin la arbitrariedad del binning ([Dimitriadis, Gneiting & Jordan](https://arxiv.org/abs/2008.03033)). Para resultados discretos (goles, posición) se usa el PIT no aleatorizado de Czado, Gneiting y Held ([Biometrics 2009](https://doi.org/10.1111/j.1541-0420.2009.01191.x)). No se encontró un análisis CORP publicado en fútbol. Para comparar modelos se usa Diebold–Mariano o Giacomini–White sobre diferencias de score, con varianzas agrupadas por fecha y bloques de temporada ([Diebold & Mariano 1995](https://doi.org/10.1080/07350015.1995.10524599); [Giacomini & White 2006](https://doi.org/10.1111/j.1468-0262.2006.00718.x)). En pronósticos de temporada hay de hecho **una sola realización independiente por liga-temporada**, así que los 20 scores de una temporada forman un clúster.

El protocolo es *walk-forward* con LFO-CV, porque LOO es optimista en series de tiempo ([Bürkner, Gabry & Vehtari](https://arxiv.org/abs/1902.06281)). Los pronósticos de temporada se emiten en checkpoints (pretemporada y fechas 5, 10, 19 y 30), y lo que se compara es la *habilidad relativa a una línea base en el mismo checkpoint*. Para combinar modelos en el régimen M-abierto se usa **stacking** con pesos que maximizan el log-score fuera de muestra, no BMA ([Yao et al. 2018](https://doi.org/10.1214/17-BA1091)). Dos advertencias aquí: toda mezcla lineal no trivial de pronósticos calibrados queda *descalibrada por falta de nitidez* ([Ranjan & Gneiting 2010](https://doi.org/10.1111/j.1467-9868.2009.00726.x)), y los pesos iguales suelen ganar cuando los pesos se estiman con pocos datos ([Claeskens et al. 2016](https://doi.org/10.1016/j.ijforecast.2015.12.005)).

El benchmark es el mercado sin margen. En el modelo de Shin, con $\pi_i=1/o_i$ y $\Pi=\sum\pi_i$,

$$
p_i(z)=\frac{\sqrt{z^2+4(1-z)\pi_i^2/\Pi}-z}{2(1-z)},\qquad \sum_ip_i(z)=1 ,
$$

con iteración de Jullien–Salanié $z\leftarrow[\sum_i\sqrt{z^2+4(1-z)\pi_i^2/\Pi}-2]/(n-2)$. El caso de prueba conocido es que las cuotas [2,6; 2,4; 4,3] dan $z\approx0{,}01694$ ([mberk/shin](https://www.github.com/mberk/shin)). Whelan sostiene que $z$ mide sobre todo el margen y no la proporción de *insiders* ([Whelan 2025](https://karlwhelan.com/sports-betting-insider-trading/)). Conviene reportar al menos dos métodos para quitar el margen, porque un modelo que le gana a las cuotas "normalizadas" puede estar ganándole solo a la normalización. Kelly fraccional es la regla de apuesta más robusta en 32.000 partidos ([Uhrín et al. 2021](https://arxiv.org/abs/2107.08827)), pero el retorno es una métrica impropia y muy ruidosa, así que debe quedar como métrica secundaria e ilustrativa.

## Arquitectura integrada: un núcleo bayesiano y seis extensiones

La pieza unificadora es un único **modelo jerárquico dinámico de goles en el grafo global**. El estado de cada equipo es $\theta_{i,t}=(\alpha_{i,t},\delta_{i,t})$, descompuesto en un efecto de liga $\lambda_{\ell,t}$ (paseo aleatorio lento) más una desviación con paseo aleatorio en días, reversión OU entre temporadas y suma cero proyectada. El prior inicial depende del valor de mercado y del ascenso. La observación es Dixon–Coles o Poisson bivariada (una sola, elegida por LFO). La simulación es predictiva posterior y respeta los comparadores por liga. Todo lo demás se acopla al núcleo como verosimilitud adicional, como prior o como postproceso de las simulaciones, de modo que el artículo puede contar una sola historia: **un campo latente gaussiano sobre un grafo dinámico, observado mediante procesos de conteo**.

| Componente | Rol | Estado | Qué lo hace interesante |
|---|---|---|---|
| M0. Paseo aleatorio + OU, Poisson con τ de Dixon–Coles, localía jerárquica | Núcleo | **Núcleo v1** | Dualidad entre decaimiento exponencial y paseo aleatorio; conservación de masa de τ |
| M1. Simulación predictiva posterior con reglas reales, CRN y estratificación | Salida Opta-style | **Núcleo v1** | Ley de varianza total; error Monte Carlo en dos niveles |
| M2. Evaluación: log/RPS/Brier, CORP, LFO, Shin | Rigor | **Núcleo v1** | $W_1$ impropia frente a CRPS propia |
| M3. Filtro EKF "Elo de goles" para actualizaciones rápidas | Operación | **Núcleo v1** | Elo = SGD = EKF de varianza fija |
| E1. Efectos de liga + aristas de transferencias y valor de mercado | Comparación entre ligas | Extensión 1 (cuando entren las copas) | **Resistencia efectiva, Fiedler, Rayleigh: probablemente original** |
| E2. Término bilineal de estilos $u_i^\top v_j$ | Head-to-head | Extensión 2 | Induce *blade-chest*; gauge $\mathrm{GL}(K)$; **sin precedentes fuera de muestra** |
| E3. Hodge en flujos de goles multi-liga con nulo calibrado | Diagnóstico de intransitividad | Extensión 2 | Componente armónica no trivial entre ligas |
| E4. Medición xG Gamma–Poisson | Más información | Extensión (solo top 5) | La precisión $\kappa$ cuantifica cuánto informa el xG |
| E5. Importancia (información mutua, $W_1$), entropía de suspenso, PLE de clasificación y número mágico | Producto/UI | Extensión prioritaria | NP-completitud con 3-1-0 |
| E6. En vivo: CTMC con rojas + Kolmogorov; Hawkes en tiros como prueba | En vivo | Extensión tardía | Concavidad, EDP de valoración, contraste de contagio |
| E7. BOCPD con hazard informado por cambios de DT / innovaciones Student-t | Robustez | Opcional | Sin aplicación publicada en fútbol |

Hay cinco contribuciones con aspecto de originales, ordenadas de mayor a menor. (1) La teoría eléctrica de identificabilidad entre ligas que combina aristas de partidos, transferencias y valor de mercado en un solo GMRF. (2) El término bilineal de estilos con la prueba de que induce un modelo intransitivo sobre resultados, contrastado fuera de muestra contra un nulo por permutación. (3) La descomposición de Hodge de flujos de goles en el grafo multi-liga, calibrada contra el ruido Poisson. (4) Una demostración empírica de la sobreconfianza del plug-in frente a la predictiva posterior en matrices de posiciones. (5) La importancia como información mutua estimada con números aleatorios comunes, más un CORP aplicado a pronósticos de temporada. En ninguno de los cinco casos se encontró trabajo previo, aunque la búsqueda bibliográfica no fue exhaustiva.

El stack de Python verificado a octubre de 2026 es el siguiente. **JAX 0.11.2 + NumPyro 0.22.0** como motor principal: `scan` para los estados, `vmap` para las ligas, NUTS/SVI. **BlackJAX 1.7.1** para SMC templado o de posteriores parciales, MCLMC y Pathfinder. **ArviZ 1.3.0**, ya basado en `xarray.DataTree`, para R-hat, ESS y PSIS. **dynamax 1.0.3** para EKF/UKF. Stan es la alternativa más madura: **CmdStan 2.40** con `sum_to_zero_vector` y Laplace embebido vía **cmdstanpy 1.3.0** (sin release desde 2025). **PyMC 6.3.2** con nutpie y Numba es la tercera vía. `particles` 0.4 (Chopin) sirve como implementación de referencia de SMC², aunque no tiene release desde 2023 ([PyPI JAX](https://pypi.org/project/jax/); [NumPyro](https://pypi.org/project/numpyro/); [BlackJAX](https://pypi.org/project/blackjax/); [ArviZ](https://pypi.org/project/arviz/); [dynamax](https://pypi.org/project/dynamax/); [PyMC releases](https://github.com/pymc-devs/pymc/releases); [particles](https://pypi.org/project/particles/)). El simulador de temporadas conviene escribirlo vectorizado en NumPy/JAX para goles y resultados, con los comparadores de desempate en Python compilado (Numba). El laplaciano y la resistencia efectiva se calculan con `scipy.sparse`, y los PLE de clasificación con un solver MIP abierto como HiGHS (versión no verificada). Para ingesta, `soccerdata` 1.9.1, que sigue mantenido aunque sus scrapers son frágiles ([soccerdata](https://soccerdata.readthedocs.io/en/latest/)).

## Las fuentes de datos de 2026 obligan a un modelo que funcione sin xG

| Necesidad | Fuente primaria | Respaldo | Advertencia 2026 |
|---|---|---|---|
| Top 5 + UCL + Brasil, resultados/fixtures | football-data.org gratis (10 req/min) ([pricing](https://www.football-data.org/pricing)) | openfootball (CC0, actualización diaria 05:00 UTC) ([repo](https://github.com/openfootball/football.json)) | Marcadores "delayed"; sin xG |
| Chile, resultados/fixtures | API-Football (gratis 100/día; Pro US$19/mes) ([pricing](https://www.api-football.com/pricing)) | ESPN `chi.1`, no oficial ([ESPN](https://espndeportes.espn.com/football/stats/_/league/chi.1)) | Chile **no** está en football-data.co.uk, openfootball ni StatsBomb; en football-data.org solo en tier pago |
| Historia + cuotas | football-data.co.uk: top 5 desde 1993/94; BRA/ARG desde 2012/13 ([data.php](https://www.football-data.co.uk/data.php)) | AvgC*/MaxC* | **El CSV E0 2026/27 no trae columnas Pinnacle (PS\*)** ([E0 2627](https://www.football-data.co.uk/mmz4281/2627/E0.csv)); no hay cuotas históricas de Chile |
| xG top 5 | Understat vía soccerdata ([understat](https://understat.com/)) | xG propio entrenado con StatsBomb open data | **FBref perdió todo el xG de Opta el 20-ene-2026** ([The IX](https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/)) |
| xG Chile/Brasil | ninguna gratuita y limpia en términos de servicio | Sportmonks xG (~€24/mes) ([pricing](https://www.sportmonks.com/football-api/plans-pricing)) | FotMob prohíbe scraping y bots ([ToS](https://www.fotmob.com/tos.txt)) |
| Valores de mercado | transfermarkt-datasets (CC0) ([repo](https://github.com/dcaribou/transfermarkt-datasets)) | scraping esporádico | **Congelado desde julio de 2026** (valoraciones hasta el 12-jun-2026) |
| Elo externo | Elo propio | ClubElo web | **API de ClubElo con 502 desde sep-2026** ([issue #977](https://github.com/probberechts/soccerdata/issues/977)) |

Las consecuencias para el modelo son directas. El núcleo debe ser **solo con goles**, y el xG debe entrar como verosimilitud *opcional* allí donde exista, que es justamente lo que permite la formulación de medición Gamma–Poisson. El benchmark de mercado debe usar Pinnacle de cierre hasta 2025/26 y, desde 2026/27, Betfair Exchange o el promedio de cierre del mercado sin margen. Para Chile no hay benchmark de mercado gratuito, por lo que la evaluación chilena se apoya en scores propios frente a líneas base. El valor de mercado de 2026/27 debe tratarse como faltante, con el prior de la temporada anterior propagado por la transición OU. Una tabla de equivalencias de nombres e IDs entre fuentes es imprescindible. La ingesta puede correr cada 30–60 minutos en ventanas de partido y disparar el reajuste y la simulación solo cuando entran resultados finales. A la fecha, la temporada chilena 2026 va en curso (Colo-Colo lideraba con 54 puntos tras 24 fechas al 11 de octubre) ([Wikipedia](https://en.wikipedia.org/wiki/2026_Liga_de_Primera)), y el Brasileirão termina el 2 de diciembre.

## Preguntas abiertas y puntos que verificar al implementar

| # | Punto | Por qué importa / cómo verificar |
|---|---|---|
| 1 | $\xi$ original de Dixon–Coles (0,0065 por media semana) y $\hat\rho$ original | No verificados; leer el paper de 1997 |
| 2 | Ecuaciones EM exactas de la Poisson bivariada | Derivación propia; contrastar con el artículo de JSS de 2005 |
| 3 | No unicidad de Sklar para marginales discretas (Genest & Nešlehová 2007) | Base teórica de la sección de copulas; no leído |
| 4 | Sub-dispersión *condicional* de los goles y valores de $\upsilon$ (CMP) | Solo hay un ajuste incondicional; estimar con datos propios |
| 5 | Identificabilidad conjunta de $(\rho,\omega,\upsilon)$ | Probable identificación débil; usar una sola corrección |
| 6 | Ganancia cuantitativa del xG sobre los goles | Sin fuente; medirla con LFO en el top 5 |
| 7 | Inicialización de los equipos ascendidos | Sin comparación publicada; probar prior con valor de mercado frente a "faltante" |
| 8 | Constantes de los pi-ratings, notación del SKF y detalles de Ingram (2021) | Marcadas [K]; leer los papers |
| 9 | Factibilidad de NUTS (~53.000 latentes) en minutos | Estimación, no medición; hacer benchmark NumPyro vs Stan vs nutpie |
| 10 | Conteo real de aristas UEFA, CONMEBOL y Mundial de Clubes por par de ligas, y resultados completos del MdC 2025 | La tabla de 12 partidos CONMEBOL–UEFA viene de memoria; calcularlo desde los fixtures |
| 11 | SE de ~0,4 goles para Premier vs Chile | Inferencia gruesa; calcular $R_{\mathrm{eff}}$ real |
| 12 | Calibración de $\kappa$ (transferencias → fuerza) y fuente de transferencias con rendimiento para Chile y Brasil | El dataset de ElHabr excluye Chile |
| 13 | Orden de desempate en Chile (¿DG antes que victorias?), Bases ANFP 2026 | Solo hay notas de Wikipedia |
| 14 | Reglamento REC 2026 del Brasileirão y cascada de cupos de la Copa do Brasil | Se asumió el orden de 2025 |
| 15 | Serie A 2026-27 (CU 244) y empates de 3 o más en 1.º y 17.º | El CU 244 no fue leído |
| 16 | Argentina: fórmula de promedios, cruce de octavos y "último en ambas tablas" | Necesario antes de incorporar la liga |
| 17 | Fórmulas originales de Geenens (2014), Schilling, Scarf–Shi y la "leverage" de FiveThirtyEight | Normalizaciones no verificadas |
| 18 | Sobreconfianza del plug-in en matrices de posiciones | Sin evidencia publicada; demostrarla con backtest |
| 19 | Si las columnas Pinnacle faltan en *todas* las ligas 2026/27 de football-data.co.uk | Solo se revisó E0, y mediante un resumidor; descargar los CSV |
| 20 | Temporadas incluidas en el plan gratuito de API-Football e IDs de liga (Chile 265, Brasil 71) | Probar con una clave gratuita |
| 21 | Umbral $\hat k$ de LFO-CV y nombres de API en ArviZ 1.x (`compare(method="stacking")`) | Verificar en la documentación |
| 22 | Coeficientes de Titman et al. (2015), Vecer et al. (2009) y Dixon–Robinson (paywall) | Afectan los priors del modelo en vivo |
| 23 | Ajuste de un Hawkes bivariado a goles con $\alpha$ reportado | No se encontró; sería un resultado nuevo, probablemente nulo |
| 24 | Cambio del pico de tiempo añadido después de 2022 | Afecta $g(t)$ en vivo; sin cuantificación publicada |

## Conclusión

El hallazgo de fondo es que, en este problema, la elegancia y la precisión casi no compiten. Como los modelos razonables de goles quedan separados por milésimas de RPS, el valor está en la *estructura* y no en exprimir la precisión. Esa estructura consiste en propagar bien la incertidumbre (la predictiva posterior frente al plug-in cambia las colas en órdenes de magnitud), aplicar bien las reglas (en Brasil la tasa de empates decide desempates) y conectar bien las ligas, donde la incertidumbre está dominada por un puente de una decena de partidos. Visto así, la comparación entre ligas deja de ser un añadido y pasa a ser el problema matemático central. Tiene una respuesta cerrada en términos de resistencia efectiva, explica por qué los priors de mercado son *necesarios* y no cosméticos, y ofrece la figura más convincente del artículo: el vector de Fiedler separando continentes.

Hay además una tensión que conviene aprovechar en el texto, no esconder. Varias ideas atractivas, como los estilos latentes, la intransitividad y el contagio tipo Hawkes, tienen evidencia previa débil o nula en goles de fútbol. En lugar de un riesgo, eso es la oportunidad de que el artículo informe *pruebas* bien diseñadas, con nulos calibrados, LFO-CV y reglas propias, cuyo resultado tiene valor aunque sea negativo. Un "no hay contagio en goles, sí en tiros" o un "la interacción de estilos mejora el log-score en 0,00x bits con IC que incluye cero" son conclusiones matemáticamente honestas y difíciles de encontrar publicadas. A eso se suma que los datos de 2026 se degradaron (sin xG en FBref, sin Pinnacle, Transfermarkt congelado), y un modelo que funciona con goles y usa lo demás como verosimilitud opcional es también el más robusto en la práctica.
