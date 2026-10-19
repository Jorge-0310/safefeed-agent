# Datos

El entrenamiento de producción de **SafeFeed-Agent** utiliza exclusivamente el dataset real **Jigsaw Toxic Comment Classification Challenge**. El corpus sintético creado en `safefeed/data.py` se utiliza únicamente en las pruebas automáticas para validar el ciclo del agente de forma rápida y reproducible; no se utiliza para reportar el desempeño real del clasificador.

## Dataset Jigsaw Toxic Comment Classification

1. Acepta las reglas de la competencia en:
   <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge>

2. Descarga los archivos del dataset. Con la CLI de Kaggle puedes ejecutar:

   ```bash
   kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
   ```

3. Descomprime el archivo descargado y coloca los siguientes archivos en esta carpeta (`data/`):

   ```text
   data/
   ├── train.csv
   ├── test.csv
   └── test_labels.csv
   ```

   El archivo `train.csv` debe contener, como mínimo, las columnas:

   ```text
   id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
   ```

   El archivo `test.csv` debe contener las columnas `id` y `comment_text`. El archivo `test_labels.csv` debe contener `id` y las etiquetas de evaluación correspondientes.

4. Verifica que el archivo de entrenamiento esté ubicado en:

   ```text
   data/train.csv
   ```

5. Entrena y evalúa el clasificador con:

   ```bash
   python -m safefeed.train
   ```

   Este comando divide `train.csv` en un conjunto de entrenamiento y otro de prueba, entrena el modelo únicamente con la partición de entrenamiento y guarda las métricas en:

   ```text
   reports/offline_metrics.json
   ```

   También intenta evaluar el modelo con `test.csv` y `test_labels.csv` cuando ambos archivos están disponibles.

## Entrenamiento más rápido

Para realizar pruebas de desarrollo sin procesar todas las filas de `train.csv`, puedes limitar el número de registros:

```bash
python -m safefeed.train --max-rows 40000
```

Este modo sirve para reducir el tiempo de entrenamiento, pero las métricas obtenidas con una muestra reducida deben identificarse como resultados de desarrollo. Para el resultado final, se recomienda entrenar con todo `train.csv`.

## Comportamiento cuando falta el dataset

Si `data/train.csv` no existe, el entrenamiento se detiene e informa que falta el dataset Jigsaw.
