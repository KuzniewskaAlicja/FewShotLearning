import torch
from torchvision import models


class ConvNeXtModel(torch.nn.Module):
    """
    A custom ConvNeXt model adapted for product classification.
    """
    def __init__(self, class_nb: int):
        super(ConvNeXtModel, self).__init__()
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        features_dim = self.backbone.classifier[0].normalized_shape[0]
        self.backbone.classifier = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LayerNorm(features_dim),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(features_dim, features_dim // 2),
            torch.nn.LayerNorm(features_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(features_dim // 2, class_nb)
        )

    def forward(self, input: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            input:
                Batch of input images with shape
                [batch_size, channels, height, width]
        """
        output = self.backbone(input)
        output = self.head(output)

        return output

    def freeze_backbone(self):
        """
        Freezes all backbone parameters while keeping the head trainable
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_model(self, unfreeze_mode: str):
        """
        Selectively unfreezes parts of the model based on the specified mode

        Args:
            unfreeze_mode:
                Specifies which parts of the model to unfreeze
        """
        if unfreeze_mode == "full":
            for param in self.parameters():
                param.requires_grad = True
        if unfreeze_mode == "last_stage":
            for param in self.parameters():
                param.requires_grad = False

            for param in self.head.parameters():
                param.requires_grad = True

            for name, param in self.backbone.named_parameters():
                if 'features.7' in name:
                    param.requires_grad = True
