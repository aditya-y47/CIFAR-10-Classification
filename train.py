import os
import warnings

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics.functional import multiclass_accuracy
from tqdm.rich import tqdm

from config import cnn_model_config, resnet_model_config, vit_model_config
from data import testloader, trainloader
from model import CNN, ResNet_tuned, VitClassfier

warnings.filterwarnings("ignore")


def train_model(model, loss_fn, trainloader, testloader, **kwargs):
    NUM_EPOCHS = kwargs.get("NUM_EPOCHS")
    BATCH_SIZE = kwargs.get("BATCH_SIZE")
    # LOG_INTERVAL = kwargs.get("LOG_INTERVAL", 1)
    LOG_INTERVAL = len(trainloader) // 10
    # VALIDATION_INTERVAL = kwargs.get("VALIDATION_INTERVAL", 2)
    VALIDATION_INTERVAL = NUM_EPOCHS // 1
    MLFLOW_MODEL_REPO = kwargs.get("MLFLOW_MODEL_REPO", "./mlflow")
    TORCH_EXPORT_PATH = kwargs.get("TORCH_EXPORT_PATH", None)
    RUN_NAME = kwargs.get("RUN_NAME", "dummy_run")
    DEVICE = kwargs.get("DEVICE", "cpu")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

    params = {
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LOG_INTERVAL": LOG_INTERVAL,
        "VALIDATION_INTERVAL": VALIDATION_INTERVAL,
        "MLFLOW_MODEL_REPO": MLFLOW_MODEL_REPO,
        "TORCH_EXPORT_PATH": TORCH_EXPORT_PATH,
        "RUN_NAME": RUN_NAME,
        "DEVICE": DEVICE,
        "OPTIMIZER": optimizer,
        "SCHEDULER": scheduler,
    }
    run_description = f"The model was trained for {params['NUM_EPOCHS']} epochs with a batch size of {params['BATCH_SIZE']}. Logging occurred every {params['LOG_INTERVAL']} batches and validation occurred every {params['VALIDATION_INTERVAL']} epochs. The trained model was saved to {params['MLFLOW_MODEL_REPO']} and exported to {params['TORCH_EXPORT_PATH']}. The run was executed on {params['DEVICE']} using the name '{params['RUN_NAME']}' in MLflow."

    mlflow.set_tracking_uri(f"file:{MLFLOW_MODEL_REPO}")
    with mlflow.start_run(run_name=RUN_NAME, description=run_description):
        mlflow.log_params(params)
        model.to(DEVICE)  # Sending the Model to DEVICE  for training
        for epoch in tqdm(range(NUM_EPOCHS), desc="Training in progress....."):
            # Training
            model.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if LOG_INTERVAL > 0 and i % LOG_INTERVAL == LOG_INTERVAL - 1:
                    avg_loss = running_loss / LOG_INTERVAL
                    print(
                        f"Epoch [{epoch+1}/{NUM_EPOCHS}]\tIteration [{i+1}/{len(trainloader)}]\tAvg. Mini-Batch Loss: {avg_loss:.5f}"
                    )
                    mlflow.log_metric("train_loss", running_loss / LOG_INTERVAL, step=i)
                    running_loss = 0.0

            # Validation
            model.eval()
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                    outputs = model(images)
                    val_loss = loss_fn(outputs, labels)
                    accuracy = multiclass_accuracy(input=outputs, target=labels)
            print("Finished Validation")
            print(f" Validation accuracy: {accuracy:.3%} ".center(50, "+"))
            mlflow.log_metric("validation_accuracy", accuracy * 100, step=epoch)
            # Learning rate scheduler step
            scheduler.step(val_loss)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch)

        if os.path.isdir(TORCH_EXPORT_PATH):
            pass
        else:
            os.mkdir(TORCH_EXPORT_PATH)

        try:
            torch.save(model.state_dict(), f"{TORCH_EXPORT_PATH}.pt")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")


if __name__ == "__main__":
    cnn_model = CNN()
    resnet_model = ResNet_tuned(num_clases=10, freeze_layers=True)
    vit_model = VitClassfier(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    # first_run
    print(f" Model Training Started for {cnn_model.__class__.__name__} ".center(50, "+"))
    train_model(
        model=cnn_model,
        loss_fn=loss_fn,
        trainloader=trainloader,
        testloader=testloader,
        **cnn_model_config,
    )
    print(f" Model Training Finished for {cnn_model.__class__.__name__} ".center(50, "+"))

    # second_run
    print(f" Model Training Started for {resnet_model.__class__.__name__} ".center(50, "+"))

    train_model(
        model=resnet_model,
        loss_fn=loss_fn,
        trainloader=trainloader,
        testloader=testloader,
        **resnet_model_config,
    )
    print(f" Model Training Finished for {resnet_model.__class__.__name__} ".center(50, "+"))

    # third_run
    print(f" Model Training Started for {vit_model.__class__.__name__} ".center(50, "+"))

    train_model(
        model=vit_model,
        loss_fn=loss_fn,
        trainloader=trainloader,
        testloader=testloader,
        **vit_model_config,
    )
    print(f" Model Training Finished for {vit_model.__class__.__name__} ".center(50, "+"))
