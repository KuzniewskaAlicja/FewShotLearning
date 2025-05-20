from typing import Any
import torch

from src.evaluation import accuracy, mean_per_class_accuracy, f1


@torch.no_grad
def validate(
    model: Any,
    val_loader: Any,
    criterion: Any,
    device: torch.device,
    topk: int = 1,
    get_acc_per_class: bool = False
) -> dict[str, float]:
    """
    Evaluates a model on the validation dataset and calculate
    performance metrics

    Args:
        model:
            Model to evaluate
        val_loader:
            Pytorch dataloader for validation data
        criterion:
            Loss function to calculate validation loss
        device:
            Device to run evaluation on (CPU or GPU)
        topk:
            Top-k accuracy to compute (default: 1)
        get_acc_per_class:
            Whether to compute and return accuracy values per class
    """
    model.eval()
    running_loss = 0.0
    acc, m_acc_per_class = 0.0, 0.0
    f1_macro, f1_micro, f1_weighted = 0.0, 0.0, 0.0
    correct_samples = torch.zeros(
        len(val_loader.dataset.classes), device=device
    )
    samples_per_class = torch.zeros(
        len(val_loader.dataset.classes), device=device
    )

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc += accuracy(outputs, targets, topk=topk)
        accuracy_per_class_results = mean_per_class_accuracy(
            outputs,
            targets,
            len(val_loader.dataset.classes),
            topk=topk,
            get_acc_per_class=get_acc_per_class
        )

        if get_acc_per_class:
            m_acc_per_class += accuracy_per_class_results[0]
            correct_samples += accuracy_per_class_results[1]
            samples_per_class += accuracy_per_class_results[2]
        else:
            m_acc_per_class += accuracy_per_class_results
        f1_macro += f1(outputs, targets, average="macro")
        f1_micro += f1(outputs, targets, average="micro")
        f1_weighted += f1(outputs, targets, average="weighted")
        running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_acc = acc / len(val_loader)
    val_mean_per_class_acc = m_acc_per_class / len(val_loader)
    val_acc_per_class = correct_samples / (samples_per_class + 1e-8)
    val_f1_macro = f1_macro / len(val_loader)
    val_f1_micro = f1_micro / len(val_loader)
    val_f1_weighted = f1_weighted / len(val_loader)

    return {
        "loss": val_loss,
        "accuracy": val_acc,
        "mean_acc_per_class": val_mean_per_class_acc.item(),
        "acc_per_class": val_acc_per_class,
        "f1_macro": val_f1_macro,
        "f1_micro": val_f1_micro,
        "f1_weighted": val_f1_weighted
    }
