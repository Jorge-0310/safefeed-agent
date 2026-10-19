# SafeFeed-Agent — Agente de moderación de contenido

Fase 2 — Desarrollo del agente inteligente  
Segundo parcial del curso de Inteligencia Artificial  
Universidad Tecnológica de Bolívar

**Integrantes:** Jorge Eliecer Martelo (T00078391) y Anderson Ibañez (T00078386)

SafeFeed-Agent es un agente de **aprendizaje con componente basado en utilidad** que, antes de publicar un comentario, estima su toxicidad y decide entre:

- Bloquearlo.
- Ocultarlo para revisión humana.
- Publicarlo.

El agente también puede mejorar mediante las decisiones confirmadas o revertidas por los moderadores simulados y mediante auditorías posteriores de publicaciones.

El clasificador de producción se entrena exclusivamente con el dataset **Jigsaw Toxic Comment Classification Challenge**, compuesto por comentarios de Wikipedia en inglés. La plataforma de comentarios, la API de publicación y los moderadores son simulados porque no se dispone de una plataforma real para realizar las pruebas.

## Inicio rápido

### 1. Crear el entorno virtual

En Linux o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 3. Preparar Jigsaw

Descarga `train.csv` desde Kaggle y colócalo en `data/train.csv`. Las instrucciones detalladas se encuentran en [`data/README.md`](data/README.md).

El archivo `train.csv` es obligatorio para entrenar el modelo de producción. El entrenamiento no sustituye automáticamente Jigsaw por el corpus sintético. El corpus sintético solo se utiliza en las pruebas automáticas.

### 4. Entrenar y evaluar el clasificador

```bash
python -m safefeed.train
```

Este comando:

1. Lee `data/train.csv`.
2. Divide los datos en 70 % para entrenamiento, 15 % para validación y 15 % para prueba.
3. Entrena candidatos con regresión logística original y con `class_weight="balanced"`.
4. Selecciona el candidato y el umbral usando únicamente validation.
5. Evalúa una sola vez el candidato elegido sobre test.
6. Guarda el modelo y las métricas en `models/` y `reports/`.

### 5. Ejecutar el agente

```bash
python -m safefeed.runner --steps 3000 --seed 7
```

### 6. Ejecutar las pruebas automáticas

```bash
python -m pytest -q
```

Las pruebas automáticas utilizan un corpus sintético pequeño y controlado para comprobar la lógica del agente. Ese corpus no se utiliza para reportar el desempeño real del clasificador.

## Salidas generadas

Los comandos generan o actualizan los siguientes archivos:

```text
models/toxicity_model.joblib
reports/offline_metrics.json
reports/last_run_summary.json
reports/policy_calibration.json
logs/run_<fecha>.jsonl
```

- `toxicity_model.joblib`: modelo entrenado.
- `offline_metrics.json`: selección, calibración y evaluación final del clasificador Jigsaw.
- `last_run_summary.json`: métricas del agente y del simulador.
- `policy_calibration.json`: barrido de políticas de utilidad en validation simulada.
- `run_<fecha>.jsonl`: registro detallado de una ejecución.

## Ablaciones para la Fase 3

Para estudiar el aporte de cada componente, pueden ejecutarse estas variantes:

```bash
python -m safefeed.runner \
  --steps 3000 \
  --seed 7 \
  --no-learning \
  --tag sin_aprendizaje
```

```bash
python -m safefeed.runner \
  --steps 3000 \
  --seed 7 \
  --no-coordination \
  --tag sin_coordinacion
```

Para iteraciones rápidas puede usarse una muestra de desarrollo:

```bash
python -m safefeed.train --max-rows 40000
```

Las métricas obtenidas con `--max-rows` deben identificarse como resultados de desarrollo y no como resultados finales del modelo completo.

El runner reentrena por defecto cada 200 etiquetas nuevas y permite modificarlo con:

```bash
python -m safefeed.runner --steps 3000 --retrain-every 200
```

## Datos y evaluación del clasificador

Para preparar Jigsaw:

1. Acepta las reglas de la competencia en <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge>.
2. Descarga `train.csv`.
3. Copia el archivo en `data/train.csv`.
4. Ejecuta `python -m safefeed.train`.

El archivo debe contener estas columnas:

```text
id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
```

El entrenamiento comprueba que la fuente sea Jigsaw y genera el reporte en `reports/offline_metrics.json`. El conjunto oficial `test.csv` de Kaggle no se utiliza para las métricas finales cuando sus etiquetas están ocultas o tienen el valor `-1`. La evaluación principal se realiza sobre la partición interna de test, separada antes de entrenar y no utilizada para seleccionar el modelo ni el umbral.

### Preparación y división del dataset

- **Limpieza de texto** (`safefeed/text.py`): minúsculas, normalización Unicode, eliminación de URLs, reducción de letras repetidas y colapso de espacios.
- **Vectorización** (`safefeed/model.py`): TF-IDF de palabras de tamaño 1–2 y de caracteres de tamaño 2–5.
- **División reproducible:** 70 % train / 15 % validation / 15 % test, estratificada por `toxic`, con semilla 42 por defecto.
- **Ausencia de fuga:** los vectores TF-IDF se ajustan con train; validation y test no se utilizan para ajustar el vectorizador ni para entrenar.
- **Selección:** validation compara el modelo original y el modelo con `class_weight="balanced"`, y calibra los umbrales de 0,10 a 0,70.
- **Evaluación final:** test se utiliza una sola vez después de seleccionar el modelo y el umbral.
- **Métricas:** se reportan precisión, recall, F1, exactitud, AUC-ROC, AUC-PR, matriz de confusión y tasas de falsos positivos y negativos.

Debido al desbalance de clases, la exactitud siempre se compara con la línea base que predice “no tóxico” para todos los comentarios.

## Dónde está el ciclo percepción → decisión → acción

El ciclo principal está implementado en la clase `SafeFeedAgent`, ubicada en [`safefeed/agent.py`](safefeed/agent.py).

| Fase | Método | Función |
|---|---|---|
| Percepción | `perceive()` | Limpia el texto, lo tokeniza y calcula señales de coordinación con la memoria de corto plazo. |
| Decisión | `decide()` | Combina toxicidad, subcategorías, coordinación y utilidad esperada. |
| Acción | `act()` | Ejecuta la acción mediante la API simulada, con reintentos y fallback seguro a revisión. |
| Aprendizaje | `receive_feedback()` y `retrain()` | Incorpora feedback de moderadores y auditorías posteriores. |
| Ciclo completo | `step()` | Encadena percepción, decisión y acción y registra cada fase. |

El agente recibe el texto, el usuario, la hora, el hilo y las señales calculadas a partir de comentarios recientes. No recibe directamente la etiqueta verdadera durante la decisión.

## Coherencia con el tipo de agente

SafeFeed-Agent es un agente de aprendizaje con componente basado en utilidad y memoria de corto plazo.

### Aprendizaje

El modelo se reentrena con ejemplos confirmados o revertidos por moderadores y con auditorías posteriores. Cada ejemplo conserva la clasificación observada como TP, TN, FP o FN. El buffer de feedback es acotado, pondera más los ejemplos recientes y cada reentrenamiento incrementa la versión del modelo.

### Utilidad

La función de utilidad está implementada en [`safefeed/utility.py`](safefeed/utility.py):

```text
EU(a) = p_ef · U(a | dañino) + (1 − p_ef) · U(a | benigno)
```

El agente compara las acciones `block`, `hide_for_review` y `publish` y escoge la de mayor utilidad esperada. `p_ef` combina la probabilidad de toxicidad con la señal de coordinación. La severidad aumenta cuando se detectan amenazas, odio identitario o toxicidad severa.

También existen barreras configurables para forzar una respuesta segura ante amenazas explícitas, toxicidad severa de alta confianza, odio identitario grave y ataques coordinados de alta confianza.

## Coordinación, fallas y logs

- La memoria usa una ventana móvil de 120 segundos simulados.
- La similitud se calcula mediante Jaccard sobre tokens distintivos.
- Se cuentan usuarios distintos y ráfagas de una misma cuenta.
- La coordinación por sí sola no escala comentarios benignos; debe existir señal de toxicidad.
- Las entradas inválidas fallan de forma cerrada a `hide_for_review`.
- La API se reintenta y, si `publish` o `block` no se ejecutan, se intenta `hide_for_review` como fallback seguro.
- Los errores se registran dentro del evento `action` mediante `attempts`, `api_ok`, `status` y `error`.
- Los logs utilizan `text_hash` SHA-256 en vez del texto completo.
- `ground_truth` aparece separado y solo lo genera el simulador o el feedback del moderador para calcular métricas.

Cada ejecución genera un archivo JSON Lines con eventos como:

```text
run_start
percept
decision
action
ground_truth
feedback
retrain
run_end
```

## Los siete componentes de la arquitectura

| Componente | Implementación |
|---|---|
| Percepción y manejo de entradas | `SafeFeedAgent.perceive()`, `text.py` y TF-IDF en `model.py`. |
| Planificación y descomposición | `SafeFeedAgent.step()`: toxicidad → coordinación → riesgo → acción. |
| Memoria | `ShortTermMemory` conserva comentarios recientes; el modelo conserva datos etiquetados y feedback. |
| Razonamiento y decisión | `SafeFeedAgent.decide()` y `utility.py`. |
| Llamada de herramientas | `SafeFeedAgent.act()` llama a `PublicationAPI.execute()`. |
| Comunicación | El agente se comunica con la API y con la cola de revisión humana. |
| Aprendizaje y adaptación | `receive_feedback()`, `retrain()` y `ToxicityModel.update()`. |

## Entorno simulado

El entorno se implementa en [`safefeed/simulator.py`](safefeed/simulator.py). La plataforma, los usuarios, las cuentas coordinadas, los tiempos, la API y los moderadores son simulados.

| Propiedad | Implementación |
|---|---|
| Parcialmente observable | El agente recibe texto y metadatos, pero no la etiqueta verdadera. |
| Estocástico | Hay llegadas aleatorias, fallos de API, errores de moderadores y tiempos variables. |
| Secuencial | Las decisiones dependen de comentarios recientes. |
| Dinámico | Aparecen oleadas coordinadas y deriva de vocabulario. |
| Discreto | El agente elige entre bloquear, ocultar o publicar. |
| Multiagente | Participan usuarios, bots coordinados, moderadores y plataforma. |

Los comentarios del simulador son frases sintéticas diseñadas para representar toxicidad, amenazas, escritura evasiva, ataques coordinados y deriva de vocabulario. No son comentarios originales de Jigsaw. Por eso, la evaluación offline del clasificador y la evaluación del agente en el simulador son resultados diferentes.

## Qué hace el código y qué hacen las herramientas externas

| Parte | Responsable |
|---|---|
| Ciclo percepción → decisión → acción | Código propio en Python |
| Memoria, coordinación y utilidad | Código propio |
| Simulador, métricas y aprendizaje | Código propio |
| Vectorización TF-IDF | scikit-learn |
| Regresión logística | scikit-learn |
| Lectura del dataset | pandas |
| Persistencia del modelo | joblib |
| Modelos de lenguaje | No se utiliza ningún LLM |

Ninguna decisión del agente depende de un modelo de lenguaje externo.

## Cambios respecto al diseño inicial

| Cambio | Justificación |
|---|---|
| La plataforma, la API y los moderadores son simulados | No se dispone de una plataforma real y el enunciado permite construir un simulador. |
| Los comentarios del simulador son sintéticos | Jigsaw no contiene oleadas, usuarios coordinados, tiempos ni deriva de vocabulario. |
| Se utiliza división 70/15/15 | Se añadió validation para seleccionar el modelo y calibrar el umbral sin utilizar test. |
| La latencia se reporta como medida del prototipo | No existe una línea base real de moderación humana para afirmar una reducción de tiempo. |
| Se añadieron auditorías posteriores | Permiten detectar algunos falsos negativos publicados y alimentar el aprendizaje. |
| Se añadieron `--no-learning` y `--no-coordination` | Permiten realizar ablaciones en la Fase 3. |

## Resultados y umbrales de aceptación

Los valores medidos se guardan en `reports/offline_metrics.json` y `reports/last_run_summary.json`.

### Clasificador Jigsaw

Resultados documentados de la ejecución completa con semilla 42:

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

El F1 supera el objetivo inicial de 0,75. El reporte también conserva si los objetivos de FNR y FPR no pueden cumplirse simultáneamente.

### Política y simulación

La política se puede calibrar con:

```bash
python -m safefeed.calibrate --steps 3000 --seed 7
```

El barrido se guarda en `reports/policy_calibration.json`. La evaluación final del agente se ejecuta con:

```bash
python -m safefeed.runner --steps 3000 --seed 17 --no-learning
```

### Batería formal de escenarios C1–C11

Ejecutar cinco repeticiones con semillas 1–5 y generar resultados vinculados a evidencia:

```bash
python -m safefeed.scenarios --seeds 1 2 3 4 5
```

El comando genera `reports/scenario_results.json` y un JSONL de evidencia en `logs/`. Cada caso
registra entrada o plantilla reproducible, condición ambiental, acción y resultado esperados,
semilla, número de repetición, resultado observado, hash del texto e índices de línea de los
eventos `percept`, `decision`, `action`, `ground_truth` y `scenario_result`.

| Código | Escenario | Acción o resultado esperado | Métrica principal |
|---|---|---|---|
| C1 | Comentario benigno normal | Publicar | Falsos bloqueos |
| C2 | Insulto claro | Ocultar o bloquear; nunca publicar | Dañinos publicados |
| C3 | Amenaza explícita | Ocultar o bloquear; nunca publicar | Dañinos publicados |
| C4 | Ataque a identidad | Ocultar o bloquear; nunca publicar | Dañinos publicados |
| C5 | Insulto con leetspeak | Ocultar o bloquear; nunca publicar | Dañinos publicados |
| C6 | Sarcasmo ambiguo benigno | Publicar o revisar; no bloquear | Falsos bloqueos |
| C7 | Texto benigno de 65.000 caracteres | Límite 20.000 caracteres; error y revisión segura sin inferencia | p95 de latencia |
| C8 | Texto vacío y espacios | Registrar error y enviar a revisión | Manejo seguro de fallas |
| C9 | Usuario/hilo ausente y timestamp NaN | Registrar cada error y enviar a revisión | Manejo seguro de fallas |
| C10 | API siempre indisponible | Reintentar, registrar y no publicar inseguro | Manejo seguro de fallas |
| C11 | Oleada de 25 comentarios coordinados | Contener ≥90 % | Contención de ataques |

C1–C10 se repiten cinco veces cada uno; C8 y C9 incluyen dos y tres subcasos por repetición,
respectivamente. C11 ejecuta cinco oleadas, una por semilla. Esta sección es una **suite funcional y
de seguridad**: sus repeticiones demuestran que se recorren las rutas previstas, pero no son textos
nuevos ni observaciones independientes y no se usan para afirmar generalización o tasas reales de
falsos positivos.

La **suite de desempeño** se informa por separado:

1. El clasificador se evalúa una sola vez sobre las **23.936 filas del test Jigsaw reservado completo**
   del split estratificado seed 42 de `train.py`. No se seleccionan manualmente textos, no se
   recalibra el umbral 0,35 y no se elige el modelo usando test. Se calcula un IC bootstrap
   estratificado de F1 (500 remuestras con predicciones y umbral fijos); se declara que alcanza
   el objetivo únicamente si el límite inferior del IC del 95 % también es ≥0,75.
2. Se ejecuta un desafío curado de 60 textos **únicos**: 30 benignos difíciles y 30 dañinos difíciles.
   Los textos se fijan en `safefeed/scenarios.py` antes de evaluarlos, una vez cada uno. Se reportan
   por dificultad y categoría; no se filtran según la probabilidad observada ni se usan para calibrar.
   Las etiquetas son de referencia, redactadas manualmente para casos sintéticos, no anotaciones de
   moderadores ni etiquetas Jigsaw.

El desafío etiqueta cada comentario con `difficulty`, `provenance` y `category`. Incluye citas de
insultos, negación, desacuerdo, menciones académicas, crítica política/sarcástica y temas sensibles
benignos; además amenazas veladas, errores ortográficos, leetspeak, palabras separadas, sustitución
Unicode/emojis, lenguaje codificado, ataques contextuales y ejemplos multilingües. Ningún caso
desafío incluye una acción esperada que fuerce la decisión del agente. Las acciones se miden después
de ejecutar la política normal. Cada resultado enlaza con los eventos del log por `comment_id` y hash.

Las tasas del conjunto desafío incluyen numerador, denominador e intervalo Wilson al 95 %. El intervalo
es descriptivo: al ser un conjunto curado pequeño, no representa una muestra aleatoria de tráfico real.
Para comparar el umbral de dañinos publicados y falsos bloqueos, se reportan tanto la estimación
puntual como el límite de confianza conservador; `pass` exige que ese límite alcance el objetivo.
La carga de revisión y la publicación automática se muestran por separado, sin un umbral añadido
retroactivamente.

El límite de comentario de 20.000 caracteres es una restricción explícita de seguridad/rendimiento
en `AgentConfig.max_comment_chars`. Las entradas más largas (C7) se envían a revisión y se registran;
no se rechazan silenciosamente ni se pasan a inferencia TF-IDF.

El escenario C7 somete deliberadamente una entrada mucho mayor que la habitual; si provoca picos de
latencia, se reportan sin excluirlos. C8–C10 contabilizan cada falla individual, incluidos los
subcasos, sin retirar fallas que hayan terminado en revisión.

#### Resultados medidos

Resultados del test Jigsaw completo y del conjunto desafío del reporte generado el 2026-10-19
(modelo y umbral originales fijos):

| Medida | n | Resultado | Incertidumbre / interpretación |
|---|---:|---:|---|
| Jigsaw test: F1 / precisión / recall | 23.936 | 0,7759 / 0,6906 / 0,8854 | IC bootstrap 95 % F1: [0,7647; 0,7854]; cumple F1 ≥0,75 |
| Jigsaw test: accuracy / AUC-ROC / AUC-PR | 23.936 | 0,9510 / 0,9806 / 0,8977 | Test completo, prevalencia tóxica 9,58 % |
| Jigsaw test: matriz `[[TN, FP], [FN, TP]]` | 23.936 | `[[20732, 910], [263, 2031]]` | FPR 4,20 %; FNR 11,46 % |
| Desafío: F1 clasificador / F1 acción final | 60 | 0,6182 / 0,7105 | 30 benignos y 30 dañinos únicos; conjunto curado, no test Jigsaw |
| Dañinos publicados en desafío | 30 dañinos | 3/30 = 10,0 % | IC Wilson 95 % [3,46 %; 25,62 %]; el límite superior no demuestra ≤10 % |
| Falsos bloqueos benignos en desafío | 30 benignos | 6/30 = 20,0 % | IC Wilson 95 % [9,51 %; 37,31 %]; no cumple ≤5 % |
| Benignos enviados a revisión | 30 benignos | 13/30 = 43,3 % | Distinto de falsos bloqueos |
| Carga de revisión total / publicación automática | 60 | 27/60 = 45,0 % / 14/60 = 23,3 % | Las 19 acciones restantes fueron bloqueos |
| Casos de desafío en banda de probabilidad 0,30–0,70 | 60 | 6 | Reportados por ID; la banda no se usó para seleccionar ni calibrar |
| Contención / manejo seguro C11 y C8–C10 | 5 repeticiones | 100 % / 100 % | Media, DE 0, mínimo y máximo se conservan por semilla |
| Latencia C1–C11 | 190 acciones | Media 3,029 ms; mediana 3,073; p95 4,571; máximo 10,950; DE 1,679 | p95 por semilla: media 4,414 ms; DE 0,165; mín. 4,220; máx. 4,641 |

El desglose por dificultad deja visible el compromiso: en los 30 casos de frontera benignos el agente
bloqueó 6 (20,0 %), revisó 13 (43,3 %) y publicó 11 (36,7 %); en los 30 adversariales dañinos publicó
3 (10,0 %), revisó 14 (46,7 %) y bloqueó 13 (43,3 %). El F1 por subgrupo de dificultad no se debe
interpretar cuando el grupo tiene una sola clase; el JSON conserva el tamaño/clase y las métricas por
categoría. El clasificador es fuerte en el test Jigsaw general, pero el conjunto curado expone
debilidades en contexto, evasión y falsos bloqueos que los escenarios fáciles no detectaban.

El resultado por subcategoría Jigsaw usa el mismo test completo y umbral diagnóstico fijo 0,50 (el
umbral 0,35 sigue siendo el de la etiqueta `toxic`); las etiquetas raras tienen pocos positivos:

| Subcategoría | Positivos en test | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| severe_toxic | 229 | 0,3636 | 0,6987 | 0,4783 |
| obscene | 1.279 | 0,7841 | 0,8577 | 0,8193 |
| threat | 76 | 0,4184 | 0,5395 | 0,4713 |
| insult | 1.186 | 0,6585 | 0,8179 | 0,7296 |
| identity_hate | 215 | 0,4056 | 0,6698 | 0,5053 |

En la suite funcional, los resultados de cada semilla, media, desviación estándar, mínimo y máximo
siguen apareciendo en el JSON. Las cinco oleadas C11 prueban contención de coordinación sintética;
no se deben mezclar con AUC-PR ni con el test Jigsaw. La latencia es del prototipo en el entorno de
ejecución, no una comparación contra moderadores humanos. No se generó video; la evidencia disponible
es el reporte y el JSONL enlazado por comentario.

Las medidas principales son:

| Medida | Umbral inicial | Campo de reporte |
|---|---:|---|
| F1 del clasificador | ≥ 0,75 | `classifier_test.metrics.f1` y `f1_95_percent_bootstrap_ci` |
| Dañinos publicados | ≤ 10 % | `challenge_set.overall.harmful_published_rate` |
| Falsos bloqueos benignos | ≤ 5 % | `challenge_set.overall.false_block_rate` |
| Benignos enviados a revisión | Diagnóstico; sin umbral fijado | `challenge_set.overall.benign_sent_to_review_rate` |
| Carga humana de revisión | Diagnóstico | `challenge_set.overall.human_review_workload` |
| Publicación automática | Diagnóstico | `challenge_set.overall.automatic_publication_rate` |
| Contención de oleadas C11 | ≥ 90 % | `acceptance.attack_containment_rate` |
| Latencia del prototipo | p95 ≤ 100 ms | `acceptance.latency_p95_ms` |
| Manejo seguro de fallas | ≥ 95 % | `acceptance.safe_failure_handling_rate` |

La latencia corresponde al entorno de ejecución del prototipo y no es una comparación contra moderadores humanos. Los resultados del simulador utilizan comentarios sintéticos y no deben presentarse como desempeño real en producción.

## Limitaciones conocidas

- El modelo se entrena con comentarios en inglés de Wikipedia y no tiene soporte multilingüe.
- Los comentarios del simulador no son los comentarios originales de Jigsaw.
- La deriva de vocabulario puede reducir el desempeño.
- Las auditorías cubren una muestra de publicaciones; los falsos negativos no auditados siguen sin etiqueta.
- El reentrenamiento reconstruye el modelo completo y puede tardar más con todo Jigsaw.
- El dataset original puede contener sesgos que este prototipo no mitiga completamente.
- Los logs almacenan hashes del texto y no el texto completo, pero en una plataforma real también deberían aplicarse políticas de retención y control de acceso.

## Estructura del proyecto

```text
safefeed/
├── agent.py          # Ciclo percepción → decisión → acción
├── utility.py        # Función de utilidad
├── coordination.py   # Memoria y coordinación
├── model.py          # Modelo de toxicidad
├── simulator.py      # Entorno simulado
├── data.py           # Carga de Jigsaw y datos de prueba sintéticos
├── text.py           # Limpieza y tokenización
├── metrics.py        # Métricas
├── logger.py         # Registro de ejecución
├── train.py          # Entrenamiento y evaluación
├── runner.py         # Ejecución del agente
├── calibrate.py      # Calibración de política
├── config.py         # Configuración
└── schemas.py        # Estructuras de datos

tests/
├── conftest.py
├── test_agent_simulator_log.py
├── test_text_utility_coordination.py
└── test_performance_safety.py

scripts/
└── validate_dataset  # Validación del dataset Jigsaw

examples/
└── sample_run.jsonl  # Ejemplo de log

data/ models/ logs/ reports/  # Entradas y salidas regenerables
```

## Referencias

- Russell, S. J., & Norvig, P. *Artificial Intelligence: A Modern Approach*. Global Edition.
- IBM. “Components of AI agents”.  
  <https://www.ibm.com/mx-es/think/topics/components-of-ai-agents>
- Jigsaw / Google. “Jigsaw Toxic Comment Classification Challenge”.  
  <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge>
- Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. “Deceiving Google’s Perspective API Built for Detecting Toxic Comments”.  
  <https://arxiv.org/abs/1702.08138>
