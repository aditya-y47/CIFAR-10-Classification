import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet152_Weights, ViT_B_16_Weights, resnet152, vit_b_16


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

    def forward(self, x):  # FORWARD FUNCTION FOR PROPOGATION OF INPUT DATA THROUGH THE NETWORK
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = self.pool(F.selu(self.conv3(x)))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = self.pool(F.selu(self.conv6(x)))

        x = x.reshape(x.shape[0], -1)  # print(x.shape)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_layer_embeds(self, num, x):
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


class ResNet_tuned(nn.Module):
    def __init__(self, num_clases: int = 10, freeze_layers: bool = True) -> None:
        super().__init__()
        self.num_classes = num_clases
        self.base_model = resnet152(weights="DEFAULT", progress=True)
        self.base_model.fc = nn.Linear(2048, 512)
        self.pp = ResNet152_Weights.IMAGENET1K_V2.transforms()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Dropout(p=0.25),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Dropout(p=0.33),
            nn.Linear(128, 32),
            nn.SELU(),
            nn.Linear(32, self.num_classes),
        )
        if freeze_layers:
            for name, param in self.base_model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x):
        x = self.pp(x)
        x = self.base_model(x)
        x = self.fc(x)
        return x


class VitClassfier(nn.Module):
    def __init__(self, num_classes: int = 10, freeze_layers: bool = True) -> None:
        super().__init__()
        self.base_model = vit_b_16(weights="DEFAULT")
        self.pp = ViT_B_16_Weights.DEFAULT.transforms()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(768, 2**8),
            nn.SELU(),
            nn.Dropout(0.33),
            nn.Linear(2**8, 2**5),
            nn.SELU(),
            nn.Linear(2**5, self.num_classes),
        )
        if freeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def feature_extractor(self, img):
        feats = nn.Sequential(*list(self.base_model.children())[:-1])
        _, encoder = feats[0], feats[1]
        x = self.base_model._process_input(img)
        cls_token = self.base_model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = encoder(x)[:, 0]
        return x

    def forward(self, x):
        x = self.pp(x)
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
