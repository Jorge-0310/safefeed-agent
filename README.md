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
respectivamente. C11 ejecuta cinco oleadas, una por semilla. Para las cinco mediciones de F1 se
utiliza el modelo y umbral ya fijados y cinco particiones estratificadas muestreadas del holdout test
reservado del split seed 42. No se recalibran pesos ni umbrales con estas particiones. Como las
particiones proceden del mismo holdout, se solapan entre semillas y no son observaciones estadísticas
independientes; se reportan como repetibilidad, no como cinco nuevos conjuntos Jigsaw independientes.

Los criterios se fijan antes de la batería: F1 ≥0,75; dañinos publicados ≤10 %; falsos bloqueos
≤5 %; contención de ataques ≥90 %; latencia p95 ≤100 ms; manejo seguro de fallas ≥95 %. Para cada
métrica el reporte incluye los cinco resultados, media, desviación estándar poblacional, mínimo y
máximo. Para F1, el umbral se evalúa sobre la media de las cinco particiones de test, mientras cada
partición se muestra y se señala cualquier valor individual bajo 0,75. Para dañinos publicados,
falsos bloqueos, contención, latencia y fallas deben pasar tanto la media como cada repetición. La
batería mide más de 50 acciones en total combinando C1–C11 a través de las cinco semillas, y no
compara latencia con moderadores humanos.

El límite de comentario de 20.000 caracteres es una restricción explícita de seguridad/rendimiento
en `AgentConfig.max_comment_chars`. Las entradas más largas (C7) se envían a revisión y se registran;
no se rechazan silenciosamente ni se pasan a inferencia TF-IDF.

El escenario C7 somete deliberadamente una entrada mucho mayor que la habitual; si provoca picos de
latencia, se reportan sin excluirlos. C8–C10 contabilizan cada falla individual, incluidos los
subcasos, sin retirar fallas que hayan terminado en revisión.

#### Resultados medidos en cinco repeticiones

Resultados de `reports/scenario_results.json` (modelo/umbral fijos; semillas 1–5):

| Criterio | Resultados por semilla | Media | DE | Mín. | Máx. | Evaluación |
|---|---|---:|---:|---:|---:|---|
| F1 Jigsaw (particiones test) | 0,7765; 0,7418; 0,7670; 0,7860; 0,7945 | 0,7732 | 0,0182 | 0,7418 | 0,7945 | **Cumple por media**; una partición queda bajo 0,75 |
| Dañinos publicados C2–C5/C11 | 0; 0; 0; 0; 0 % | 0 % | 0 % | 0 % | 0 % | **Cumple media y cada repetición** |
| Falsos bloqueos C1/C6 | 0; 0; 0; 0; 0 % | 0 % | 0 % | 0 % | 0 % | **Cumple media y cada repetición** |
| Contención C11 | 100; 100; 100; 100; 100 % | 100 % | 0 % | 100 % | 100 % | **Cumple media y cada repetición** |
| p95 latencia por semilla (ms) | 4,532; 4,936; 4,480; 5,199; 4,769 | 4,783 | 0,265 | 4,480 | 5,199 | **Cumple media y cada repetición** |
| Manejo seguro C8–C10 | 100; 100; 100; 100; 100 % | 100 % | 0 % | 100 % | 100 % | **Cumple media y cada repetición** |

La F1 se acepta con el umbral predefinido sobre la **media**, no se oculta la variación: la semilla 2
dio 0,7418. Las cinco particiones se muestrean del mismo holdout reservado y se solapan; son una
prueba de sensibilidad a la partición, no cinco conjuntos independientes. Para las métricas de
seguridad y operación se requiere que tanto media como todas las semillas pasen.

En las 190 acciones de escenarios, la latencia total tuvo media 2,979 ms, mediana 2,758 ms, p95
4,943 ms, máximo 11,001 ms y DE 1,739 ms. C7, de 65.000 caracteres, fue a revisión por el límite
de entrada; su latencia media fue 10,393 ms (DE 0,450; mín. 9,615; máx. 11,001). C8 gestionó
espacios/vacío, C9 sus tres metadatos inválidos y C10 cuatro intentos API fallidos por repetición;
los tres escenarios acabaron en revisión y registraron la falla. C6 se envió a revisión en las cinco
repeticiones: no cuenta como falso bloqueo, pero sí es una carga de revisión del 100 % para ese caso.

Las cinco oleadas C11 contuvieron 25/25 comentarios cada una. C1 publicó el benigno; C2 y C3
bloquearon; C4 y C5 ocultaron para revisión; C6 y C7 ocultaron para revisión; C8–C10 usaron la
acción segura esperada. El log JSONL contiene eventos por comentario y referencias de línea en el
reporte. No se generó captura de video ni minuto de video; la evidencia verificable disponible es
el JSONL y el reporte estructurado.

Las medidas principales son:

| Medida | Umbral inicial | Campo de reporte |
|---|---:|---|
| F1 del clasificador | ≥ 0,75 | `final_test.f1` |
| Comentarios dañinos publicados | ≤ 10 % | `agent.harmful_published_rate` |
| Falsos bloqueos benignos | ≤ 5 % | `agent.false_block_rate` |
| Contención de ataques | ≥ 90 % | `simulator.attacks.contained_rate` |
| Latencia del prototipo | p95 ≤ 100 ms | `agent.latency_ms_p95` |
| Manejo seguro de fallas | ≥ 95 % | `failure_handling.safe_failure_handling_rate` |

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
