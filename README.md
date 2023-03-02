# Performance Study of CV Architectures

This project aims to study the performance of various CV architectures, including Vanilla CNN, CNN with transfer learning, and ViT, on standard dataset like cifar-10. The project is moslty written in `PyTorch` and utilizes a trainer code and logs the results to an MLFlow directory called `mlruns`.

## Installation and Usage

First, clone this repository to your local machine use the develop branch

```sh
git clone -b develop https://github.com/pyro99X/CIFAR-10-Classification.git
cd CIFAR-10-Classification
#  activate venv or conda and run the shell script to begin training and export the artifacts to a zip file at root/
#  Requires zip cli too be be installed
sh ./run.sh
```

```sh
# Only training the models
python3 train.py
```

Read throguh the configs listed in `./config.py`, these configs are dynamically loaded by the training code
I have run the training jobs and exported the models [here](https://www.dropbox.com/sh/6n7zs1hkaa6fcmu/AACNSWQ-C37EfpDIFTR1Fb3Ua?dl=0)

download and uncompress these file and place them in `./models/torch_models` in case you'd like to test the models by performing inference.

## ML Flow

To view the results in MLFlow without trainig, navigate to the mlruns directory and run the following command:

```sh
# This will open a web browser displaying the MLFlow dashboard.
mlflow ui
```
