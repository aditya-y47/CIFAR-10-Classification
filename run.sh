# /bin/bash
pip install -U pip
pip install mlflow rich tqdm torcheval
python train.py
zip -q -9 -r ./exported_models.zip ./models
zip -q -9 -r ./exported_mlflow_runs.zip ./mlruns
