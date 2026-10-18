# SafeFeed-Agent — Agente de moderación de contenido

Fase 2 — Desarrollo del agente inteligente segundo parcial del curso de Inteligencia Artificial, Universidad Tecnológica de Bolívar.

**Integrantes:** [Jorge Eliecer Martelo (T00078391) y Anderson Ibañez (T00078386)]

Implementa el diseño de **SAFEFEED-AGENT**: un agente **de aprendizaje con componente basado en
utilidad** que, antes de publicar un comentario, estima su toxicidad y decide entre
**bloquear**, **ocultar para revisión humana** o **publicar**, y mejora con las decisiones
confirmadas o revertidas por moderadores.

El clasificador se entrena con el dataset **Jigsaw Toxic Comment Classification Challenge** (Kaggle/Google,
comentarios de Wikipedia en inglés). La plataforma de comentarios, la API de publicación y los
moderadores son simulados (ver la sección "Entorno simulado").

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Descargar train.csv de Jigsaw y copiarlo en data/train.csv (ver "Datos")
python -m safefeed.train                  # entrena el clasificador y mide en un conjunto de prueba
python -m safefeed.runner --steps 3000 --seed 7   # corre el agente contra la plataforma simulada
python -m pytest -q                       # 26 pruebas automáticas
```

Requiere Python 3.10+ (en nuestro caso, probado con 3.13).

Salidas: `models/toxicity_model.joblib`, `reports/offline_metrics.json`,
`reports/last_run_summary.json` y **`logs/run_<fecha>.jsonl`** (el log de ejecución).

Ablaciones útiles para la Fase 3:

```bash
python -m safefeed.runner --steps 3000 --seed 7 --no-learning --tag sin_aprendizaje
python -m safefeed.runner --steps 3000 --seed 7 --no-coordination --tag sin_coordinacion
```

Si el reentrenamiento tarda demasiado con el dataset completo, se puede usar
`python -m safefeed.train --max-rows 40000` y `--retrain-every 200` en el runner.

## Datos

1. Aceptar las reglas de la competencia en
   <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge> y descargar `train.csv`.
2. Copiarlo en `data/train.csv` (columnas: `id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`).
3. Ejecutar `python -m safefeed.train`. En la salida, `"data_source"` debe decir `"jigsaw"`.

`train.csv` no se versiona porque está en .gitignore, por lo que cada persona debe descargarlo. Sin ese archivo el entrenamiento se detiene con un error, porque no hay corpus de respaldo. Las pruebas automáticas (pytest) no necesitan el CSV porque usan un conjunto mínimo de ejemplos propio.


### Preparación de los datos

- **Limpieza de texto** (`safefeed/text.py`): minúsculas, normalización Unicode, eliminación de URLs,
  reducción de letras repetidas (3 o más pasan a 2) y colapso de espacios.
- **Vectorización** (`safefeed/model.py`): TF-IDF de palabras (1–2-gramas) y de caracteres (2–5-gramas,
  que dan algo de robustez ante escritura evasiva como "1d10t").
- **División:** 80 % entrenamiento / 20 % prueba, estratificada por la etiqueta `toxic`, de modo que ambos
  conjuntos conservan la misma proporción de comentarios tóxicos. El conjunto de prueba no se usa para entrenar.
- **Desbalance de clases:** los comentarios tóxicos son una minoría
  ([COMPLETAR: % de tóxicos en train.csv, p. ej. el resultado de `df["toxic"].mean()`]). No se aplica
  remuestreo ni `class_weight`; el desbalance se maneja en la decisión, porque la función de utilidad penaliza
  más dejar pasar un comentario dañino que enviar uno benigno a revisión. Por esta razón la exactitud se
  compara siempre con la línea base "publicar siempre" y se reportan también precisión, recall, F1 y AUC-PR.

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

**Nota:** los comentarios que genera el simulador son frases de plantilla, no comentarios reales de Jigsaw.
El modelo se entrena y evalúa con Jigsaw, pero el flujo simulado es otro tipo de texto; por eso los
resultados de la simulación y los de la evaluación offline no son comparables entre sí.

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
| Los comentarios del simulador son **sintéticos** (de plantilla), aunque el modelo se entrena con Jigsaw | Hace falta un flujo con olas de ataque coordinado y deriva de vocabulario, que Jigsaw no contiene |
| "Riesgo bajo / intermedio / grave" **no son umbrales fijos**: son las regiones donde gana cada acción en la utilidad esperada | Es la forma coherente de un agente basado en utilidad; las regiones dependen de `UtilityConfig`, de la severidad y de la ola |
| Subcategorías = las 5 de Jigsaw (`severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) | Son las etiquetas del dataset de referencia |
| La meta "reducir 50 % el tiempo de respuesta" se mide como **latencia por comentario (ms)** y como **proxy de carga humana** (`human_review_workload_reduction`: fracción que no va a revisión manual) | La Fase 1 detectó que la meta original no tenía línea base medible |
| Los moderadores simulados revisan la cola de "ocultos" y atienden apelaciones al 8 % de los bloqueos | Es la fuente de retroalimentación del diseño (confirmación/reversión) |
| Se añadieron las banderas `--no-learning` y `--no-coordination` | Permiten las ablaciones de la Fase 3 |

## Resultados (entrenamiento con Jigsaw)

**Evaluación offline** (`python -m safefeed.train`, conjunto de prueba del 20 %, umbral 0.5):

| Métrica | Valor |
|---|---|
| Comentarios de entrenamiento / prueba | [COMPLETAR] / [COMPLETAR] |
| Exactitud | [COMPLETAR] |
| Línea base "siempre no tóxico" | [COMPLETAR] |
| Precisión / Recall / F1 | [COMPLETAR] / [COMPLETAR] / [COMPLETAR] |
| AUC-ROC / AUC-PR | [COMPLETAR] / [COMPLETAR] |

Estos valores salen de `reports/offline_metrics.json`.

**Simulación** (`python -m safefeed.runner --steps 3000 --seed 7`):

| Métrica | Valor |
|---|---|
| Exactitud (línea base "publicar siempre") | [COMPLETAR] ([COMPLETAR]) |
| Precisión / Recall | [COMPLETAR] / [COMPLETAR] |
| Bloqueos injustos (`false_block_rate`) | [COMPLETAR] |
| Comentarios dañinos publicados (`harmful_published_rate`) | [COMPLETAR] |
| Ataques contenidos | [COMPLETAR] |
| Latencia media / p95 (ms) | [COMPLETAR] / [COMPLETAR] |
| Veredictos de moderadores / reentrenamientos | [COMPLETAR] / [COMPLETAR] |

Estos valores salen de `reports/last_run_summary.json`.

**Cómo leerlos:** la exactitud sola es engañosa con clases desbalanceadas, por eso se compara siempre con la
línea base y se mira también el F1 y el AUC-PR. Los resultados de la simulación miden si el agente completo
(percepción, utilidad, coordinación, aprendizaje) funciona sobre un flujo simulado, no la calidad del clasificador
sobre comentarios reales; para eso sirve la evaluación offline.

## Limitaciones conocidas

- Solo se aprende de lo que se oculta o se bloquea. Un comentario dañino que el modelo publica con
  probabilidad muy baja **nunca recibe veredicto**; así, los falsos negativos no se corrigen solos.
  Mejora futura: canal de denuncias de usuarios sobre comentarios publicados.
- Inglés únicamente (el dataset de referencia es de Wikipedia en inglés); no hay soporte multilingüe.
- Reentrenar re-ajusta todo el modelo: con Jigsaw completo (~160 000 filas) tarda varios segundos o minutos
  por reentrenamiento; usar `--max-rows` al entrenar o subir `--retrain-every`.
- El simulador no reproduce comentarios reales de Jigsaw, por lo que el modelo puede rendir distinto sobre el flujo simulado que en la evaluación offline.
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