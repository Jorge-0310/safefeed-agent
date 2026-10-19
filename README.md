# SafeFeed-Agent — Agente de moderación de contenido

Fase 2 — Desarrollo del agente inteligente  
Segundo parcial del curso de Inteligencia Artificial  
Universidad Tecnológica de Bolívar

**Integrantes:** Jorge Eliecer Martelo (T00078391) y Anderson Ibañez (T00078386)

SafeFeed-Agent es un agente de **aprendizaje con componente basado en utilidad** que, antes de publicar un comentario, estima su toxicidad y decide entre:

- Bloquearlo.
- Ocultarlo para revisión humana.
- Publicarlo.

El agente también puede mejorar mediante las decisiones confirmadas o revertidas por los moderadores simulados.

El clasificador se entrena exclusivamente con el dataset **Jigsaw Toxic Comment Classification Challenge**, compuesto por comentarios de Wikipedia en inglés. La plataforma de comentarios, la API de publicación y los moderadores son simulados porque no se dispone de una plataforma real para realizar las pruebas.

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

### 3. Preparar el dataset Jigsaw

Descarga y coloca los archivos del dataset en la carpeta `data/`:

```text
data/
├── train.csv
├── test.csv
└── test_labels.csv
```

Las instrucciones detalladas se encuentran en [`data/README.md`](data/README.md).

El archivo `train.csv` es obligatorio para entrenar el modelo de producción. El proyecto no utiliza automáticamente el corpus sintético como respaldo.

### 4. Entrenar y evaluar el clasificador

```bash
python -m safefeed.train
```

Este comando:

1. Lee `data/train.csv`.
2. Divide los datos en 80 % para entrenamiento y 20 % para prueba.
3. Entrena el clasificador con la partición de entrenamiento.
4. Evalúa el modelo sobre la partición de prueba.
5. Evalúa adicionalmente sobre `test.csv` y `test_labels.csv` si ambos archivos están disponibles.
6. Guarda el modelo y las métricas.

### 5. Ejecutar el agente

```bash
python -m safefeed.runner --steps 3000 --seed 7
```

### 6. Ejecutar las pruebas automáticas

```bash
python -m pytest -q
```

Las pruebas automáticas utilizan un corpus sintético pequeño y controlado para comprobar el comportamiento del agente. Este corpus no se utiliza para reportar el desempeño real del clasificador.

## Salidas generadas

Los comandos generan los siguientes archivos:

```text
models/toxicity_model.joblib
reports/offline_metrics.json
reports/last_run_summary.json
logs/run_<fecha>.jsonl
```

- `toxicity_model.joblib`: modelo entrenado.
- `offline_metrics.json`: métricas del clasificador sobre Jigsaw.
- `last_run_summary.json`: métricas de la última simulación.
- `run_<fecha>.jsonl`: registro detallado de la ejecución.

## Ablaciones para la Fase 3

Para estudiar el aporte de cada componente, pueden ejecutarse las siguientes variantes:

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

Estas ejecuciones permiten comparar:

- El agente completo.
- El agente sin aprendizaje por retroalimentación.
- El agente sin detección de coordinación.

Si el reentrenamiento tarda demasiado con todo el dataset, puede utilizarse una muestra de desarrollo:

```bash
python -m safefeed.train --max-rows 40000
```

En ese caso, las métricas deben identificarse como resultados de desarrollo y no como resultados finales del modelo completo.

## Datos y preparación

El entrenamiento de producción utiliza exclusivamente el dataset Jigsaw.

### Limpieza de texto

La limpieza se realiza en `safefeed/text.py` e incluye:

- Conversión a minúsculas.
- Normalización Unicode.
- Eliminación de URLs.
- Reducción de letras repetidas.
- Colapso de espacios.
- Tokenización básica.

### Vectorización

El modelo utiliza una combinación de características TF-IDF:

- N-gramas de palabras de tamaño 1 y 2.
- N-gramas de caracteres de tamaño 2 a 5.

Los n-gramas de caracteres aportan cierta robustez frente a variaciones ortográficas y escritura evasiva, como `1d10t`.

### División del dataset

El archivo `train.csv` se divide de la siguiente forma:

- 70 % para entrenamiento.
- 30 % para prueba.
- División estratificada según la etiqueta `toxic`.

El conjunto de prueba no se utiliza para entrenar el modelo.

### Desbalance de clases

Los comentarios tóxicos son una minoría dentro del dataset. Por esta razón, la exactitud se compara con una línea base que siempre predice “no tóxico”.

También se reportan:

- Precisión.
- Recall.
- F1.
- AUC-ROC.
- AUC-PR.

No se utiliza `class_weight` ni remuestreo. El desbalance se considera dentro de la función de utilidad, que penaliza con mayor fuerza dejar pasar contenido dañino que enviar un comentario benigno a revisión.

## Dónde está el ciclo percepción → decisión → acción

El ciclo principal está implementado en la clase `SafeFeedAgent`, ubicada en [`safefeed/agent.py`](safefeed/agent.py).

| Fase | Método | Función |
|---|---|---|
| Percepción | `perceive()` | Limpia el texto, lo tokeniza y calcula señales de coordinación utilizando la memoria de corto plazo. |
| Decisión | `decide()` | Combina la probabilidad de toxicidad, las subcategorías, la coordinación y la función de utilidad. |
| Acción | `act()` | Ejecuta la acción seleccionada mediante la API simulada de publicación. |
| Aprendizaje | `receive_feedback()` y `retrain()` | Reciben los veredictos de los moderadores y reentrenan el modelo periódicamente. |
| Ciclo completo | `step()` | Encadena percepción, decisión y acción y registra cada etapa en el log. |

El agente no recibe directamente la etiqueta verdadera del comentario. Solo recibe:

- El texto.
- El identificador del usuario.
- La marca de tiempo.
- El hilo de conversación.
- Las señales calculadas a partir de comentarios recientes.

La retroalimentación llega posteriormente desde los moderadores simulados.

## Coherencia con el tipo de agente

SafeFeed-Agent es un agente de aprendizaje con componente basado en utilidad y memoria de corto plazo.

### Aprendizaje

El modelo ubicado en `safefeed/model.py` se reentrena con ejemplos etiquetados por los moderadores:

```python
ToxicityModel.update()
```

Los ejemplos provenientes de la retroalimentación reciben un peso superior al de los datos originales. Esto permite que el modelo se adapte a nuevos patrones de lenguaje y a las decisiones del equipo de moderación.

### Utilidad

La función de utilidad está implementada en [`safefeed/utility.py`](safefeed/utility.py).

Para cada acción se calcula una utilidad esperada:

```text
EU(a) = p_ef · U(a | dañino) + (1 − p_ef) · U(a | benigno)
```

Donde:

- `p_ef` es la probabilidad efectiva de que el comentario sea dañino.
- `U(a | dañino)` es la utilidad de ejecutar la acción cuando el comentario es dañino.
- `U(a | benigno)` es la utilidad de ejecutar la acción cuando el comentario es benigno.

El agente compara las utilidades de:

- `block`
- `hide_for_review`
- `publish`

y selecciona la acción con mayor utilidad esperada.

La probabilidad efectiva combina la predicción de toxicidad con la señal de coordinación. Además, los casos relacionados con amenazas, odio o toxicidad severa reciben un factor de severidad superior.

Los parámetros de la función de utilidad están definidos en `safefeed/config.py`.

## Los siete componentes de la arquitectura

| Componente | Implementación |
|---|---|
| Percepción y manejo de entradas | `SafeFeedAgent.perceive()`, `text.py` y TF-IDF en `model.py`. |
| Planificación y descomposición | `SafeFeedAgent.step()`: probabilidad de toxicidad → coordinación → riesgo → acción. |
| Memoria | `ShortTermMemory` conserva comentarios recientes. El modelo conserva los datos etiquetados y el búfer de retroalimentación. |
| Razonamiento y decisión | `SafeFeedAgent.decide()` y `utility.py` calculan la utilidad esperada de cada acción. |
| Llamada de herramientas | `SafeFeedAgent.act()` llama a `PublicationAPI.execute()`. |
| Comunicación | El agente se comunica con la API simulada y envía los casos ocultos a la cola de revisión humana. |
| Aprendizaje y adaptación | `receive_feedback()`, `retrain()` y `ToxicityModel.update()`. |

## Entorno simulado

El entorno se implementa en [`safefeed/simulator.py`](safefeed/simulator.py).

El simulador genera las percepciones descritas en el diseño y permite probar el agente sin utilizar una plataforma real.

| Propiedad | Implementación |
|---|---|
| Parcialmente observable | El agente recibe texto y metadatos, pero no recibe la etiqueta verdadera. |
| Estocástico | Se generan llegadas aleatorias, fallos de API, errores de moderadores y tiempos de respuesta variables. |
| Secuencial | Las decisiones dependen de los comentarios recientes almacenados en memoria. |
| Dinámico | Aparecen oleadas coordinadas y deriva de vocabulario durante la ejecución. |
| Discreto | El agente selecciona entre bloquear, ocultar o publicar. |
| Multiagente | Participan usuarios, cuentas coordinadas, moderadores y el sistema de publicación. |

La misma semilla produce el mismo flujo de comentarios y las mismas decisiones del agente. Las marcas de tiempo reales de los logs pueden variar entre ejecuciones.

El simulador contiene frases sintéticas diseñadas para representar:

- Comentarios benignos.
- Insultos.
- Amenazas.
- Discurso de odio.
- Escritura evasiva.
- Ataques coordinados.
- Deriva de vocabulario.

Estos comentarios no son los comentarios originales de Jigsaw. Por esta razón, los resultados offline sobre Jigsaw y los resultados de la simulación no deben interpretarse como la misma métrica.

## Registro de ejecución

Cada ejecución genera un archivo JSON Lines en la carpeta `logs/`.

Cada evento contiene:

- `ts`: marca de tiempo real en UTC.
- `sim_time`: tiempo simulado.
- `run_id`: identificador de la ejecución.
- `event`: tipo de evento.

Los eventos principales son:

- `run_start`
- `percept`
- `decision`
- `action`
- `ground_truth`
- `feedback`
- `retrain`
- `run_end`

El evento `ground_truth` lo escribe exclusivamente el simulador para permitir el cálculo de métricas. El agente no recibe esta información durante la percepción ni durante la toma de decisiones.

Un ejemplo corto de log se encuentra en [`examples/sample_run.jsonl`](examples/sample_run.jsonl).

## Qué hace el código y qué hacen las herramientas externas

| Parte | Responsable |
|---|---|
| Ciclo percepción → decisión → acción | Código propio en Python |
| Memoria de corto plazo | Código propio en `coordination.py` |
| Detección de coordinación | Código propio en `coordination.py` |
| Función de utilidad | Código propio en `utility.py` |
| Selección de la acción | Código propio en `agent.py` y `utility.py` |
| Simulación del entorno | Código propio en `simulator.py` |
| Aprendizaje por retroalimentación | Código propio en `agent.py` y `model.py` |
| Vectorización TF-IDF | scikit-learn |
| Regresión logística | scikit-learn |
| Lectura del dataset | pandas |
| Persistencia del modelo | joblib |
| Modelos de lenguaje | No se utiliza ningún LLM |

Ninguna decisión del agente depende de un modelo de lenguaje externo.

## Cambios respecto al diseño inicial

| Cambio | Justificación |
|---|---|
| La plataforma, la API de publicación y los moderadores son simulados | No se dispone de una plataforma real; el enunciado permite construir un entorno simulado. |
| Los comentarios del simulador son sintéticos | Se necesitan oleadas coordinadas, escritura evasiva y deriva de vocabulario, características que no están disponibles directamente en Jigsaw. |
| Las categorías de riesgo no utilizan únicamente umbrales fijos | Las acciones se seleccionan mediante utilidad esperada, de acuerdo con el tipo de agente basado en utilidad. |
| Las subcategorías corresponden a las etiquetas de Jigsaw | Se utilizan `severe_toxic`, `obscene`, `threat`, `insult` e `identity_hate`. |
| La reducción del tiempo se mide mediante latencia y carga de revisión | La meta inicial de reducir el tiempo en 50 % no tenía una línea base suficientemente definida. |
| Los moderadores revisan los comentarios ocultos y algunas apelaciones | Esto proporciona la retroalimentación necesaria para el componente de aprendizaje. |
| Se añadieron las opciones `--no-learning` y `--no-coordination` | Permiten realizar ablaciones durante la Fase 3. |

## Resultados

### Evaluación offline sobre Jigsaw

La evaluación se ejecuta con:

```bash
python -m safefeed.train
```

El comando divide `train.csv` en una partición de entrenamiento y otra de prueba. Los resultados se guardan en:

```text
reports/offline_metrics.json
```

| Métrica | Valor |
|---|---:|
| Comentarios de entrenamiento / prueba | `[COMPLETAR] / [COMPLETAR]` |
| Tasa de toxicidad en prueba | `[COMPLETAR]` |
| Exactitud | `[COMPLETAR]` |
| Línea base “siempre no tóxico” | `[COMPLETAR]` |
| Precisión | `[COMPLETAR]` |
| Recall | `[COMPLETAR]` |
| F1 | `[COMPLETAR]` |
| AUC-ROC | `[COMPLETAR]` |
| AUC-PR | `[COMPLETAR]` |

La exactitud debe interpretarse junto con la línea base, el F1, el recall y el AUC-PR debido al desbalance entre comentarios tóxicos y no tóxicos.

Si `test.csv` y `test_labels.csv` están disponibles, también se reportan las métricas sobre el conjunto oficial de prueba de Jigsaw.

### Evaluación del agente en el entorno simulado

La simulación se ejecuta con:

```bash
python -m safefeed.runner --steps 3000 --seed 7
```

Los resultados se guardan en:

```text
reports/last_run_summary.json
```

| Métrica | Valor |
|---|---:|
| Exactitud | `[COMPLETAR]` |
| Línea base “publicar siempre” | `[COMPLETAR]` |
| Precisión | `[COMPLETAR]` |
| Recall | `[COMPLETAR]` |
| F1 | `[COMPLETAR]` |
| Bloqueos injustos (`false_block_rate`) | `[COMPLETAR]` |
| Comentarios benignos enviados a revisión | `[COMPLETAR]` |
| Comentarios dañinos publicados (`harmful_published_rate`) | `[COMPLETAR]` |
| Ataques contenidos | `[COMPLETAR]` |
| Latencia media | `[COMPLETAR] ms` |
| Latencia p95 | `[COMPLETAR] ms` |
| Veredictos de moderadores | `[COMPLETAR]` |
| Reentrenamientos | `[COMPLETAR]` |

La evaluación simulada mide el comportamiento del agente completo sobre un entorno controlado. No representa directamente la calidad del clasificador sobre comentarios reales de Jigsaw.

## Limitaciones conocidas

- El modelo se entrena únicamente con comentarios en inglés de Wikipedia.
- No se incluye soporte multilingüe.
- Los comentarios del simulador son sintéticos y no representan exactamente los comentarios de Jigsaw.
- La deriva de vocabulario puede reducir el desempeño del clasificador.
- El modelo solo recibe retroalimentación de comentarios ocultos o bloqueados; los comentarios dañinos publicados no reciben automáticamente un veredicto.
- El reentrenamiento reconstruye el modelo completo y puede tardar más cuando se utiliza todo el dataset.
- El dataset original puede contener sesgos que no se mitigan completamente en este prototipo.
- Los logs almacenan el texto de los comentarios y los identificadores simulados de usuario; en una plataforma real deberían aplicarse controles de privacidad y retención de datos.

## Estructura del proyecto

```text
safefeed/
├── agent.py          # Ciclo percepción → decisión → acción
├── utility.py        # Función de utilidad
├── coordination.py   # Memoria y coordinación
├── model.py           # Modelo de toxicidad
├── simulator.py       # Entorno simulado
├── data.py            # Carga de Jigsaw y datos sintéticos para pruebas
├── text.py            # Limpieza y tokenización
├── metrics.py         # Métricas
├── logger.py          # Registro de ejecución
├── train.py           # Entrenamiento y evaluación
├── runner.py          # Ejecución del agente
├── config.py          # Configuración
└── schemas.py         # Estructuras de datos

tests/
├── conftest.py
├── test_agent_simulator_log.py
└── test_text_utility_coordination.py

examples/
└── sample_run.jsonl

data/
├── README.md
├── train.csv
├── test.csv
└── test_labels.csv
```

## Referencias

- Russell, S. J., & Norvig, P. *Artificial Intelligence: A Modern Approach*. Global Edition.
- IBM. “Components of AI agents”.  
  <https://www.ibm.com/mx-es/think/topics/components-of-ai-agents>
- Jigsaw / Google. “Jigsaw Toxic Comment Classification Challenge”.  
  <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge>
- Hosseini, H., Kannan, S., Zhang, B., & Poovendran, R. “Deceiving Google’s Perspective API Built for Detecting Toxic Comments”.  
  <https://arxiv.org/abs/1702.08138>
