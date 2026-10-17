# Datos

Por defecto el proyecto usa un **corpus sintético de semilla** (ver `safefeed/data.py`) para que
corra sin descargar nada. Sirve para probar el agente, **no** para afirmar desempeño real.

## Usar el dataset real (Jigsaw Toxic Comment Classification)

1. Acepta las reglas de la competencia en
   <https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge> y descarga `train.csv`.
   (Con la CLI de Kaggle: `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge`).
2. Copia `train.csv` en esta carpeta: `data/train.csv`
   (columnas: `id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`).
3. Reentrena: `python -m safefeed.train` (o `python -m safefeed.train --max-rows 40000` para ir más rápido).

Si `data/train.csv` existe, el agente lo usa automáticamente; si no, usa el corpus sintético.
