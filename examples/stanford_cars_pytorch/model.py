import torchvision.models as models
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Model, self).__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 196)

    def forward(self, x):
        return self.model(x)

    # Count total number of trainable parameters in the model.
    # [NOTE] This is example helper function, not part of DeltaFabric usage.
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
