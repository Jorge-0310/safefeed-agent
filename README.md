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
python -m pytest -q                       # pruebas automáticas
```

Requiere Python 3.10+ (en nuestro caso, probado con 3.13).

Salidas: `models/toxicity_model.joblib`, `reports/offline_metrics.json`,
`reports/last_run_summary.json` y **`logs/run_<fecha>.jsonl`** (el log de ejecución).

Ablaciones útiles para la Fase 3:

```bash
python -m safefeed.runner --steps 3000 --seed 7 --no-learning --tag sin_aprendizaje
python -m safefeed.runner --steps 3000 --seed 7 --no-coordination --tag sin_coordinacion
```

Para iteraciones rápidas se puede usar `python -m safefeed.train --max-rows 40000`; el runner
reentrena por defecto cada 200 etiquetas nuevas y permite ajustarlo con `--retrain-every`.

## Datos y evaluación del clasificador

1. Aceptar las reglas de la competencia en
   <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge> y descargar `train.csv`.
2. Copiarlo en `data/train.csv` (columnas: `id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`).
3. Ejecutar `python -m safefeed.train`. En la salida, `"data_source"` debe decir `"jigsaw"`.
4. Calibrar la política en una semilla de validación simulada, distinta del test final:
   `python -m safefeed.calibrate --steps 3000 --seed 7`. El barrido y los objetivos quedan en
   `reports/policy_calibration.json`.

`train.csv` no se versiona porque está en .gitignore, por lo que cada persona debe descargarlo. El entrenamiento normal no sustituye silenciosamente un corpus Jigsaw ausente por datos sintéticos. Las pruebas automáticas usan su corpus sintético de prueba.


### Preparación de los datos

- **Limpieza de texto** (`safefeed/text.py`): minúsculas, normalización Unicode, eliminación de URLs,
  reducción de letras repetidas (3 o más pasan a 2) y colapso de espacios.
- **Vectorización** (`safefeed/model.py`): TF-IDF de palabras (1–2-gramas) y de caracteres (2–5-gramas,
  que dan algo de robustez ante escritura evasiva como "1d10t").
- **División reproducible:** 70 % train / 15 % validation / 15 % test, estratificada por `toxic`,
  con semilla fija 42 por defecto. Los vectores TF-IDF se ajustan solo con train. No se sobremuestrean
  validation ni test.
- **Desbalance:** se entrenan dos candidatos sobre el mismo train: regresión logística original y
  `class_weight="balanced"`. Validation compara los dos; test se usa una sola vez para reportar el
  candidato elegido, nunca para escoger pesos, umbral o política.
- **Calibración:** validation evalúa umbrales 0,10–0,70 en pasos de 0,05. Se prioriza FNR dañino
  ≤10 % y FPR ≤5 %; si ambos límites no son alcanzables, el reporte lo indica y hace explícito el
  compromiso. El umbral final queda persistido en el modelo y el barrido completo está en
  `reports/offline_metrics.json`.
- Se reportan precisión, recall, F1, exactitud, AUC-ROC, AUC-PR, matriz de confusión y conteos de
  falsos positivos/negativos. El objetivo inicial de F1 es ≥0,75 y no se oculta si no se alcanza.

## Dónde está el ciclo percepción → decisión → acción

Todo vive en [`safefeed/agent.py`](safefeed/agent.py), clase `SafeFeedAgent`:

| Fase | Método | Qué hace |
|---|---|---|
| Percepción | `perceive()` | Limpia el texto, tokeniza y calcula señales de coordinación con la memoria de corto plazo |
| Decisión | `decide()` | Probabilidad de toxicidad + subcategoría → probabilidad efectiva → utilidad esperada de cada acción → elige la mayor |
| Acción | `act()` | Llama a la API de publicación (bloquear / ocultar / publicar), con un reintento si falla |
| Aprendizaje | `receive_feedback()`, `retrain()` | Incluye auditorías posteriores de comentarios publicados y reentrena periódicamente con ejemplos recientes |
| Ciclo | `step()` | Encadena percepción → decisión → acción y escribe cada fase en el log |

El agente **nunca ve la etiqueta verdadera** (hay una prueba que lo verifica): solo texto y metadatos,
y después la retroalimentación de moderadores.

## Coherencia con el tipo de agente

- **Aprendizaje:** el modelo (`safefeed/model.py`) se reentrena con ejemplos confirmados/revertidos por
  moderadores y auditorías posteriores de publicaciones. Se registran TP/TN/FP/FN; un FN publicado se
  conserva en el buffer, se reentrena y se prueba el cambio posterior de probabilidad/decisión.
  El buffer es acotado, pondera más los ejemplos recientes y cada reentrenamiento incrementa la versión.
- **Utilidad:** [`safefeed/utility.py`](safefeed/utility.py) compara las tres acciones con
  `EU(a) = p_ef · U(a | dañino) + (1 − p_ef) · U(a | benigno)`, donde `p_ef` combina la
  probabilidad del modelo con la señal de coordinación y los términos "dañino" se escalan por
  severidad (amenaza, odio, toxicidad severa). Los valores están en `safefeed/config.py`
  (`UtilityConfig`). En la política elegida en validación las regiones base quedan en ~0.02
  (publicar → ocultar) y ~0.68 (ocultar → bloquear). Se calculan y registran en cada reporte.

Las barreras configurables fuerzan `block` ante amenazas explícitas y toxicidad severa de alta
confianza, `hide_for_review` ante odio identitario grave y `block` ante ataques coordinados de alta
confianza. El resto de los casos sigue usando la función de utilidad esperada.

## Coordinación, fallas y logs

- La memoria usa una ventana móvil de **120 segundos simulados** (configurable); la similitud es
  Jaccard `|A ∩ B| / |A ∪ B|` sobre tokens distintivos. Se descartan frases comunes y textos con menos
  de tres tokens distintivos; se cuentan usuarios distintos y ráfagas de una cuenta. La coordinación
  por sí sola no escala comentarios benignos: la utilidad exige señal tóxica.
- Una fracción configurable de publicaciones se audita después de publicada. Así se detectan FNs
  que nunca entrarían a la cola de revisión. El reporte separa pre/post deriva; el evento `retrain`
  compara el ajuste del buffer antes/después y registra la versión del modelo.
- Entradas inválidas, señal de coordinación sin datos y respuestas API inválidas/no disponibles
  fallan de forma cerrada a revisión. La API se reintenta; si publicar/bloquear falla se intenta
  `hide_for_review`. Los errores quedan explícitos en el log.
- Los logs usan `text_hash` SHA-256 en vez del texto y registran coordinación, probabilidades,
  utilidades, acción ejecutada, API, latencia, versión y error. `ground_truth` queda separado y
  marcado como procedente del simulador o de feedback del moderador.

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
| La latencia se mide desde percepción hasta ejecución de la acción en el prototipo (media, mediana, p95, máximo y desviación); la carga humana es el número de comentarios enviados a revisión | No existe línea base de moderación manual; no se afirma una reducción frente a moderadores humanos |
| Los moderadores simulados revisan la cola de "ocultos" y atienden apelaciones al 8 % de los bloqueos | Es la fuente de retroalimentación del diseño (confirmación/reversión) |
| Se añadieron las banderas `--no-learning` y `--no-coordination` | Permiten las ablaciones de la Fase 3 |

## Resultados y umbrales de aceptación

Los valores medidos se guardan en `reports/offline_metrics.json` y
`reports/last_run_summary.json` al ejecutar los comandos de entrenamiento y simulación. El primero
contiene la evaluación final **solo en test** y, por separado, la comparación de modelos/calibración
hecha en validation. El segundo separa `classifier`, `agent` y `simulator`; sus AUC pertenecen al
clasificador, nunca a la contención de ataques.

### Clasificador Jigsaw

Ejecución completa con seed 42; selección y calibración en validation, evaluación final en test:

| Elemento | Resultado |
|---|---:|
| División train / validation / test | 111.699 / 23.936 / 23.936 |
| Tasa tóxica train / validation / test | 9,58 % / 9,58 % / 9,58 % |
| Modelo seleccionado | `class_weight="balanced"` |
| Umbral seleccionado en validation | 0,35 |
| Accuracy / línea base siempre benigno | 95,10 % / 90,42 % |
| Precisión / recall / F1 | 69,06 % / 88,54 % / **77,59 %** |
| AUC-ROC / AUC-PR | 0,9806 / 0,8977 |
| Matriz de confusión `[[TN, FP], [FN, TP]]` | `[[20732, 910], [263, 2031]]` |

En validation, el candidato original con umbral 0,15 logró recall 85,22 % y FPR 3,84 %;
el balanceado con 0,35 logró recall 86,62 % y FPR 4,55 %. Se eligió el balanceado por mayor
recall bajo el límite FPR, aunque su F1 de validation fue menor (75,48 % frente a 76,98 %).
El F1 supera el objetivo inicial de 0,75. En validation, ningún umbral de la rejilla logró
simultáneamente FNR ≤10 % y FPR ≤5 %; se escogió el mejor recall dentro del límite FPR y el reporte
conserva ese aviso. El barrido no usa el test.

### Política y simulación final

`python -m safefeed.calibrate --steps 3000 --seed 7` seleccionó
`publish_harmful=-10.0` minimizando daño publicado entre políticas que cumplen recall ≥90 %,
falsos bloqueos ≤5 % y contención ≥90 %. La región base correspondiente es `publish` [0, 0,0231),
`hide_for_review` [0,0231, 0,6801) y `block` [0,6801, 1]; coordinación, severidad y barreras
configurables pueden cambiar la acción final.
En la semilla de validación, esta política logró 0 % de dañinos publicados, 3,24 % de falsos
bloqueos benignos y 100 % de contención; el benigno enviado a revisión fue 22,71 %.

Evaluación reproducible: `python -m safefeed.runner --steps 3000 --seed 17 --no-learning`,
3.000 comentarios sintéticos (aprendizaje desactivado para
medir la política fija; el ciclo de aprendizaje tiene pruebas específicas):

| Métrica del agente | Resultado |
|---|---:|
| Accuracy (baseline publicar siempre) | 80,43 % (79,30 %) |
| Precisión / recall / F1 de la acción del agente | 51,42 % / 99,36 % / 67,76 % |
| Dañinos publicados | 0,64 % |
| Falsos bloqueos benignos | 2,61 % |
| Benignos enviados a revisión (separado) | 21,90 % |
| Contención de ataques | 331/331 = 100 % |
| Ataques publicados antes de la primera contención | 0 |
| Latencia media / mediana / p95 / máxima / desv. estándar | 2,937 / 2,743 / 3,445 / 133,662 / 3,376 ms |
| Errores API simulados | 31; los 31 se manejaron de forma segura |
| Matriz de fallas inyectadas | 11/11 manejadas correctamente (100 %) |

La contención fue 166/166 antes de la deriva y 165/165 después; las dos campañas tuvieron 100 %
de contención (media 100 %, desviación 0, mínimo 100 %, máximo 100 %). Los benignos enviados a
revisión no se cuentan como falsos bloqueos. La latencia p95 cumple 100 ms aunque hubo un máximo
aislado de 133,662 ms. Es latencia del prototipo en este entorno de ejecución, **no** una comparación
con moderadores humanos.

Las métricas por fase están en `by_phase`; el rendimiento del clasificador en el flujo sintético
está separado del F1 Jigsaw de test. La simulación final desactivó aprendizaje, así que no se afirma
que esa corrida mida mejora adaptativa: `tests/test_performance_safety.py` verifica un falso negativo
publicado, su incorporación al buffer, reentrenamiento y cambio de probabilidad/acción; las corridas
con aprendizaje habilitado registran comparaciones por versión.

| Medida | Umbral inicial | Campo de reporte |
|---|---:|---|
| F1 del clasificador | ≥0,75 | `final_test.f1` |
| Dañinos publicados | ≤10 % | `agent.harmful_published_rate` |
| Falsos bloqueos benignos | ≤5 % | `agent.false_block_rate` |
| Benignos enviados a revisión | Informativo, separado de falsos bloqueos | `agent.benign_sent_to_review_rate` |
| Contención de ataques | ≥90 % | `simulator.attacks.contained_rate`, por campaña y fase |
| Latencia de prototipo | p95 ≤100 ms | `agent.latency_ms_p95` |
| Manejo seguro de fallas inyectadas (entrada, sensor y API) | ≥95 % | `failure_handling.safe_failure_handling_rate` |

La latencia corresponde al entorno de ejecución del prototipo, desde percepción hasta respuesta de
la acción; **no** es una comparación contra moderadores humanos. Los resultados del simulador usan
comentarios sintéticos y no se deben presentar como desempeño real en producción.

## Limitaciones conocidas

- Las auditorías simuladas cubren una muestra de publicaciones, no todas; los falsos negativos no
  auditados siguen sin etiqueta. La exactitud posterior al reentrenamiento del buffer es una medida
  de ajuste sobre feedback, no una evaluación independiente.
- Inglés únicamente (el dataset de referencia es de Wikipedia en inglés); no hay soporte multilingüe.
- Reentrenar re-ajusta todo el modelo: con Jigsaw completo (~160 000 filas) tarda varios segundos o minutos
  por reentrenamiento; usar `--max-rows` al entrenar o subir `--retrain-every`.
- El simulador no reproduce comentarios reales de Jigsaw, por lo que el modelo puede rendir distinto sobre el flujo simulado que en la evaluación offline.
- Sesgos del dataset original (comentarios de Wikipedia) no se mitigan aquí; es un riesgo ético declarado en el diseño.

## Estructura

```
safefeed/        agent.py (ciclo) · utility.py · coordination.py · model.py · simulator.py
                 data.py · text.py · metrics.py · logger.py · train.py · runner.py · config.py · schemas.py
tests/           pruebas de utilidad, coordinación, ciclo, logs, fallas, métricas, simulador y aprendizaje
examples/        sample_run.jsonl — ejemplo corto de log
data/ models/ logs/ reports/   entradas y salidas (se regeneran)
```

## Referencias

- Russell, S. J., & Norvig, P. *Artificial Intelligence: A Modern Approach* (Global Edition).
- IBM. "Components of AI agents". <https://www.ibm.com/mx-es/think/topics/components-of-ai-agents>
- Jigsaw / Google. "Jigsaw Toxic Comment Classification Challenge" (Kaggle).
- Hosseini, H., et al. "Deceiving Google's Perspective API Built for Detecting Toxic Comments" (arXiv:1702.08138) — motiva los n-gramas de caracteres.