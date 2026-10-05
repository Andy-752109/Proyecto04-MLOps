import json

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(config):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for name, param in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")):
            param.requires_grad = False
        else:
            param.requires_grad = True

    num_ftrs = model.fc.in_features

    torch.manual_seed(config.seed_model)

    if config.hidden_layers == 1:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(256, 2)
        )
    else:
        model.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(num_ftrs, 2))
    return model


def export_architecture_and_class_map(
    arch_path: str = "architecture.json", class_map_path: str = "class_map.json"
):
    with open(arch_path, "w") as f:
        json.dump(
            {"base_model": "resnet18", "pretrained": "IMAGENET1K_V1", "head": "configurable"}, f
        )
    with open(class_map_path, "w") as f:
        json.dump({"0": "cat", "1": "dog"}, f)
