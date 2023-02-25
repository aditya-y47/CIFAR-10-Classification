import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # FC Linear Layers
        self.fc1 = nn.Linear(12800, 64)
        self.fc2 = nn.Linear(64, 12)
        self.fc3 = nn.Linear(12, 10)

    def forward(
        self, x
    ):  # FORWARD FUNCTION FOR PROPOGATION OF INPUT DATA THROUGH THE NETWORK
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool(F.selu(self.conv3(x)))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = self.pool(F.selu(self.conv6(x)))

        x = x.reshape(
            x.shape[0], -1
        )  # print(x.shape)                          # TEMPORARY PRINT STATEMNETS I USED WHILE ALTERING THE NN STRUCTURE TO VERIFY THE IMAGE DIMENSTIONS AND MAINTAIN COMPATABILITY

        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_layer_embeds(
        self, num, x
    ):  # Series of Layer embeddings to help with visulization of feature maps genrated after covultion operation.
        if num == 1:
            return self.conv1(x)
        elif num == 2:
            return self.pool(self.conv1(x))
        elif num == 3:
            return self.conv2(x)
        elif num == 4:
            return self.pool(self.conv2(x))
        elif num == 5:
            return self.conv3(x)
        elif num == 6:
            return self.pool(self.conv3(x))
        elif num == 7:
            return self.conv4(x)
        elif num == 8:
            return self.pool(self.conv4(x))
        elif num == 9:
            return self.conv5(x)
        elif num == 10:
            return self.pool(self.conv5(x))
        elif num == 11:
            return self.conv6(x)
        elif num == 12:
            return self.pool(self.conv6(x))


class TransferInceptionNetV3(nn.Module):
    def __init__(self, num_clases: int = 10, freeze_layers: bool = True) -> None:
        super().__init__()
        self.num_classes = num_clases
        self.base_model = inception_v3(weights="IMAGENET1K_V1", progress=True)
        self.base_model.fc = nn.Linear(2048, 1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256), nn.SELU(), nn.Linear(256, self.num_classes)
        )

        if freeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.resize = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def forward(self, x):
        x = self.resize(x)
        x = self.base_model(x)
        x = self.fc(x)
        return x
