import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

BATCH_SIZE = 2**6
cnn_model_config = {
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": BATCH_SIZE,
    "LOG_INTERVAL": None,
    "VALIDATION_INTERVAL": None,
    "DEVICE": device,
    "RUN_NAME": "CNN-test",
    "MLFLOW_MODEL_REPO": "./mlruns",
    "TORCH_EXPORT_PATH": None,
}
cnn_model_config["TORCH_EXPORT_PATH"] = f"./models/torch_models/{cnn_model_config['RUN_NAME']}"


resnet_model_config = {
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": BATCH_SIZE,
    "LOG_INTERVAL": None,
    "VALIDATION_INTERVAL": None,
    "DEVICE": device,
    "RUN_NAME": "ResNet-test",
    "MLFLOW_MODEL_REPO": "./mlruns",
    "TORCH_EXPORT_PATH": None,
}
resnet_model_config[
    "TORCH_EXPORT_PATH"
] = f"./models/torch_models/{resnet_model_config['RUN_NAME']}"

vit_model_config = {
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": BATCH_SIZE,
    "LOG_INTERVAL": None,
    "VALIDATION_INTERVAL": None,
    "DEVICE": device,
    "RUN_NAME": "ViT-test",
    "MLFLOW_MODEL_REPO": "./mlruns",
    "TORCH_EXPORT_PATH": None,
}
vit_model_config["TORCH_EXPORT_PATH"] = f"./models/torch_models/{vit_model_config['RUN_NAME']}"
