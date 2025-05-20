from typing import Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch


def create_results_description(
    top1: dict, top5: dict, class_names: list
) -> str:
    """
    Generates a human-readable report of model evaluation results including:
    - accuracy metrics
    - F1 scores
    - the best and worst performing classes

    Args:
        top1:
            Dictionary containing top-1 evaluation metrics including:
        top5:
            Dictionary containing top-5 evaluation metrics
        class_names:
            List of class names corresponding to model outputs
    """
    metrics_desc = (
        "Results of provided model\n\n"
        f"Accuracy: Top-1 - {top1['accuracy']:.2f}, "
        f"Top-5 - {top5['accuracy']:.2f}\n"
        f"Mean accuracy per class: "
        f"Top-1 - {top1['mean_acc_per_class']:.2f} "
        f"Top-5 - {top5['mean_acc_per_class']:.2f}\n"
        f"f1-score (macro): {top1['f1_macro']:.2f}\n"
        f"f1-score (micro): {top1['f1_micro']:.2f}\n"
        f"f1-score (weighted): {top1['f1_weighted']:.2f}\n\n"
    )

    top1_acc, top1_indices = top1["acc_per_class"].sort()
    best_worst_classes = "5 Worst Performing Classes (Top-1 Accuracy)\n"
    for i in range(5):
        class_name = class_names[top1_indices[i].item()]
        best_worst_classes = (
            best_worst_classes + f"- {class_name}: {top1_acc[i].item()}\n"
        )

    best_worst_classes = (
        best_worst_classes + "\n5 Best Performing Classes (Top-1 Accuracy)\n"
    )
    for i in range(1, 6):
        class_name = class_names[top1_indices[-i].item()]
        best_worst_classes = (
            best_worst_classes + f"- {class_name}: {top1_acc[-i].item()}\n"
        )

    return metrics_desc + best_worst_classes


def denormalize_image(
    image: torch.Tensor, mean: list[float], std: list[float]
) -> torch.Tensor:
    """
    Reverts normalization applied to an image for visualization

    Args:
        image:
            Normalized image tensor
        mean:
            List of mean values used in normalization
        std:
            List of standard deviation values used in normalization
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean


@torch.no_grad
def plot_predictions(
    model: Any, data_loader: Any, samples_nb: int, device: torch.device
) -> plt.figure:
    """
    Generates a figure showing model predictions and ground truth labels
    on distinct class samples

    Args:
        model:
            Model to evaluate
        data_loader:
            Pytorch dataLoader for validation data
        samples_nb:
            Number of distinct class samples to display
        device:
            Device to run inference on (CPU or GPU)
    """
    model.eval()
    normalization_params = data_loader.dataset.transforms.transforms[-1]

    images_shown = 0
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('Example predictions', fontsize=15)
    gs = GridSpec(3, 5, figure=fig)

    plot_classes = []
    col_idx = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        for i in range(images.size(0)):
            if len(plot_classes) >= samples_nb:
                break

            pred_label = preds[i].item()
            true_label = labels[i].item()

            if true_label in plot_classes:
                continue

            img = (
                denormalize_image(
                    images[i].cpu(),
                    normalization_params.mean,
                    normalization_params.std
                )
                .permute(1, 2, 0)
                .clamp(0, 1)
                .numpy() * 255
            ).astype(np.uint8)

            title = f"Pred: {pred_label}\nGT: {true_label}"

            col_idx = col_idx if col_idx < 5 else 0
            ax = fig.add_subplot(gs[len(plot_classes) // 5, col_idx])
            ax.set_title(title)
            ax.set_axis_off()
            ax.imshow(img)

            images_shown += 1
            col_idx += 1
            plot_classes.append(true_label)

        if len(plot_classes) >= samples_nb:
            break

    plt.tight_layout()

    return fig
