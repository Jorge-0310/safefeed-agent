# SafeFeed-Agent — Agente de moderación de contenido

Fase 2 (desarrollo) del curso de Inteligencia Artificial, Universidad Tecnológica de Bolívar.
Implementa el diseño de **SAFEFEED-AGENT**: un agente **de aprendizaje con componente basado en
utilidad** que, antes de publicar un comentario, estima su toxicidad y decide entre
**bloquear**, **ocultar para revisión humana** o **publicar**, y mejora con las decisiones
confirmadas o revertidas por moderadores.

> **Importante (honestidad de los resultados).** El entorno es **simulado** y, por defecto, el
> clasificador se entrena con un **corpus sintético de semilla** (plantillas en inglés). Eso valida
> que el *agente completo funciona* (ciclo, utilidad, memoria, aprendizaje, log, pruebas), **no**
> que alcance 90 % sobre comentarios reales. Para cifras reales hay que entrenar con Jigsaw
> (ver [`data/README.md`](data/README.md)).

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m safefeed.train                  # entrena el clasificador y mide en un conjunto de prueba
python -m safefeed.runner --steps 3000 --seed 7   # corre el agente contra la plataforma simulada
python -m pytest -q                       # 26 pruebas automáticas
```

Requiere Python 3.10+ (probado con 3.13). Cada comando termina en menos de un minuto con el corpus
sintético. Si no entrenas antes, `runner` entrena el modelo automáticamente la primera vez.

Salidas: `models/toxicity_model.joblib`, `reports/offline_metrics.json`,
`reports/last_run_summary.json` y **`logs/run_<fecha>.jsonl`** (el log de ejecución).

Ablaciones útiles para la Fase 3 (misma semilla, un factor distinto):

```bash
python -m safefeed.runner --steps 3000 --seed 7 --no-learning --tag sin_aprendizaje
python -m safefeed.runner --steps 3000 --seed 7 --no-coordination --tag sin_coordinacion
```

## Dónde está el ciclo percepción → decisión → acción

Todo vive en [`safefeed/agent.py`](safefeed/agent.py), clase `SafeFeedAgent`:

| Fase | Método | Qué hace |
|---|---|---|
| Percepción | `perceive()` | Limpia el texto, tokeniza y calcula señales de coordinación con la memoria de corto plazo |
| Decisión | `decide()` | Probabilidad de toxicidad + subcategoría → probabilidad efectiva → utilidad esperada de cada acción → elige la mayor |
| Acción | `act()` | Llama a la API de publicación (bloquear / ocultar / publicar), con un reintento si falla |
| Aprendizaje | `receive_feedback()`, `retrain()` | Guarda veredictos de moderadores y reentrena cada `retrain_every` eventos |
| Ciclo | `step()` | Encadena percepción → decisión → acción y escribe cada fase en el log |

El agente **nunca ve la etiqueta verdadera** (hay una prueba que lo verifica): solo texto y metadatos,
y después la retroalimentación de moderadores.

## Coherencia con el tipo de agente

- **Aprendizaje:** el modelo (`safefeed/model.py`) se reentrena con ejemplos confirmados/revertidos por
  moderadores (`ToxicityModel.update`, peso ×3 frente a los datos originales). Hay una prueba que
  comprueba que, tras la retroalimentación, la probabilidad de un texto nuevo sube y la versión del modelo avanza.
- **Utilidad:** [`safefeed/utility.py`](safefeed/utility.py) compara las tres acciones con
  `EU(a) = p_ef · U(a | dañino) + (1 − p_ef) · U(a | benigno)`, donde `p_ef` combina la
  probabilidad del modelo con la señal de coordinación y los términos "dañino" se escalan por
  severidad (amenaza, odio, toxicidad severa). Los valores están en `safefeed/config.py`
  (`UtilityConfig`). Con esos valores, sin severidad ni ola de ataque, las regiones quedan en
  ~0.11 (publicar → ocultar) y ~0.68 (ocultar → bloquear).

## Los 7 componentes IBM del diseño y su código

| Componente | Implementación |
|---|---|
| Percepción y manejo de entradas | `SafeFeedAgent.perceive`, `text.py` (limpieza/tokenización), TF-IDF palabra + carácter en `model.py` |
| Planificación y descomposición | `SafeFeedAgent.step`: probabilidad → comparación con ataques recientes → riesgo → acción (planificación reactiva, corto horizonte) |
| Memoria | Corto plazo: `coordination.ShortTermMemory` (ventana de 120 s simulados). Largo plazo: datos etiquetados dentro de `ToxicityModel` + búfer de retroalimentación |
| Razonamiento y decisión | `SafeFeedAgent.decide` + `utility.py` (utilidad esperada, modo "ola" ante ataques coordinados) |
| Llamada de herramientas | `SafeFeedAgent.act` → `PublicationAPI.execute` (API simulada, con fallos aleatorios y reintento) |
| Comunicación | Llamadas a la API; casos intermedios a la cola de revisión humana con probabilidad y subcategoría (eventos `decision`/`action` del log); `receive_feedback` cierra el ciclo |
| Aprendizaje y adaptación | `receive_feedback`, `retrain`, `ToxicityModel.update` |

## Entorno simulado (propiedades de la sección 3 del diseño)

[`safefeed/simulator.py`](safefeed/simulator.py) genera exactamente las percepciones de la sección 5 y
respeta las propiedades declaradas:

| Propiedad | Cómo se respeta |
|---|---|
| Parcialmente observable | El agente recibe texto + usuario + hora + hilo. La verdad (`GroundTruth`) solo la usan el moderador simulado y las métricas |
| Estocástico | Llegadas exponenciales, mezcla aleatoria, fallos aleatorios de la API (1 %), moderadores que se equivocan (7 %) y tardan un tiempo aleatorio |
| Secuencial | Las olas de ataque repiten texto; detectarlas exige comparar con comentarios recientes |
| Dinámico | El flujo no espera al agente; hay 2 olas coordinadas y una deriva de vocabulario y frases a mitad de la simulación |
| Discreto | Tres acciones y subcategorías; la probabilidad continua se traduce a una de ellas |
| Multiagente | Usuarios, cuentas coordinadas (bots), moderadores y API de publicación son actores separados |

La misma semilla produce exactamente el mismo flujo y las mismas decisiones.

## Log de ejecución

Cada ejecución escribe un archivo JSON Lines (`logs/run_<fecha>.jsonl`), una línea por evento, con
`ts` (hora real UTC) y `sim_time` (reloj simulado). Eventos: `run_start`, `percept`, `decision`
(probabilidades, utilidades esperadas y rationale), `action` (resultado de la API), `ground_truth`
(**la escribe el simulador; el agente no la ve**), `feedback`, `retrain`, `run_end` (resumen de métricas).
Un ejemplo corto está en [`examples/sample_run.jsonl`](examples/sample_run.jsonl). Estos logs son la base de la Fase 3.

## Qué razona el código y qué hacen las herramientas externas

| Parte | Quién la hace |
|---|---|
| Ciclo del agente, memoria, señal de coordinación, función de utilidad, elección de acción, reentrenamiento, simulador, métricas | **Código propio** (Python) |
| Vectorización TF-IDF y regresión logística (entrenamiento e inferencia) | **scikit-learn** (el agente decide *cuándo* reentrenar y con qué ejemplos) |
| Lectura de datos y persistencia | pandas, joblib |
| **Modelos de lenguaje (LLM)** | **No se usa ninguno.** Ninguna decisión del agente depende de un LLM |

## Cambios respecto al diseño (documentados y justificados)

| Qué cambió | Por qué |
|---|---|
| Plataforma, API de publicación y moderadores son **simulados** | No hay acceso a una plataforma real; el enunciado permite un simulador que respete las propiedades del ambiente |
| El corpus de entrenamiento por defecto es **sintético** (Jigsaw es opcional) | Kaggle exige cuenta y aceptar reglas; el repo debe correr en otro equipo sin pasos manuales. Con Jigsaw (`data/train.csv`) se usa automáticamente |
| "Riesgo bajo / intermedio / grave" **no son umbrales fijos**: son las regiones donde gana cada acción en la utilidad esperada | Es la forma coherente de un agente basado en utilidad; las regiones dependen de `UtilityConfig`, de la severidad y de la ola |
| Subcategorías = las 5 de Jigsaw (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) | Son las etiquetas del dataset de referencia |
| La meta "reducir 50 % el tiempo de respuesta" se mide como **latencia por comentario (ms)** y como **proxy de carga humana** (`human_review_workload_reduction`: fracción que no va a revisión manual) | La Fase 1 detectó que la meta original no tenía línea base medible |
| Los moderadores simulados revisan la cola de "ocultos" y atienden apelaciones al 8 % de los bloqueos | Es la fuente de retroalimentación del diseño (confirmación/reversión) |
| Se añadieron las banderas `--no-learning` y `--no-coordination` | Permiten las ablaciones de la Fase 3 |

## Resultados de ejemplo (corpus sintético, semilla 7)

Offline (`python -m safefeed.train`, 6 400 train / 1 600 test): exactitud 0.978 frente a 0.886 de
"publicar siempre", F1 0.895, AUC-ROC 0.919, AUC-PR 0.862.

Simulación (`--steps 3000 --seed 7`): exactitud 0.987 (línea base "publicar siempre": 0.793),
precisión 0.942, recall 1.0, 0 % de bloqueos injustos, 100 % de los 339 comentarios de ataque
contenidos, latencia media ≈ 2.7 ms (p95 ≈ 3.8 ms), 127 veredictos de moderadores y 2 reentrenamientos.

**Cómo leerlos:** la exactitud sola es engañosa con clases desbalanceadas (siempre se compara con la línea
base). Y como los datos sintéticos salen de pocas plantillas, el problema es fácil: el AUC en simulación
da ≈ 1.0 y la diferencia entre con y sin aprendizaje es mínima (recall 1.0 frente a 0.998 en esta semilla).
Esto confirma que el pipeline funciona, no que el agente rinda así con comentarios reales.

## Limitaciones conocidas

- Solo se aprende de lo que se oculta o se bloquea. Un comentario dañino que el modelo publica con
  probabilidad muy baja **nunca recibe veredicto**; así, los falsos negativos no se corrigen solos.
  Mejora futura: canal de denuncias de usuarios sobre comentarios publicados.
- Inglés únicamente (el dataset de referencia es de Wikipedia en inglés); no hay soporte multilingüe.
- Reentrenar re-ajusta todo el modelo: con Jigsaw completo (~160 000 filas) tarda varios segundos o minutos
  por reentrenamiento; usa `--max-rows` al entrenar o sube `--retrain-every`.
- Sesgos del dataset original (comentarios de Wikipedia) no se mitigan aquí; es un riesgo ético declarado en el diseño.

## Estructura

```
safefeed/        agent.py (ciclo) · utility.py · coordination.py · model.py · simulator.py
                 data.py · text.py · metrics.py · logger.py · train.py · runner.py · config.py · schemas.py
tests/           26 pruebas (utilidad, coordinación, ciclo, log, simulador, aprendizaje, reproducibilidad)
examples/        sample_run.jsonl — ejemplo corto de log
data/ models/ logs/ reports/   entradas y salidas (se regeneran)
```

## Referencias

- Russell, S. J., & Norvig, P. *Artificial Intelligence: A Modern Approach* (Global Edition).
- IBM. "Components of AI agents". <https://www.ibm.com/mx-es/think/topics/components-of-ai-agents>
- Jigsaw / Google. "Jigsaw Toxic Comment Classification Challenge" (Kaggle).
- Hosseini, H., et al. "Deceiving Google's Perspective API Built for Detecting Toxic Comments" (arXiv:1702.08138) — motiva los n-gramas de caracteres.
