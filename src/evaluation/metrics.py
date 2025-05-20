import torch
from sklearn.metrics import f1_score


def accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: int = 1
) -> float:
    """
    Calculates standard accuracy

    Args:
        outputs:
            Model outputs (logits)
        targets:
            Ground truth labels
        topk:
            Number of top predictions to consider
    """
    _, predicted = outputs.topk(k=topk, dim=1)
    expanded_targets = targets.view(-1, 1).expand(-1, topk)
    correct = predicted.eq(expanded_targets).any(dim=1).sum().item()

    return correct / targets.size()[0]


def mean_per_class_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    topk: int = 1,
    get_acc_per_class: bool = False
) -> float | tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculates class-averaged accuracy (mean per-class accuracy)

    Args:
        outputs:
            Model outputs (logits)
        targets:
            Ground truth labels
        num_classes:
            Number of classes
        topk:
            Number of top predictions to consider
    """
    _, predicted = outputs.topk(k=topk, dim=1)

    total_classes_samples_nb = torch.zeros(
        num_classes, dtype=torch.float
    ).to(targets.device)
    correct_samples = torch.zeros(
        num_classes, dtype=torch.float
    ).to(targets.device)

    indexes = targets.type(torch.int64)
    total_classes_samples_nb = total_classes_samples_nb.scatter_add_(
        0, indexes, torch.ones_like(targets, dtype=torch.float)
    )
    expanded_targets = targets.view(-1, 1).expand(-1, topk)
    is_correct = predicted.eq(expanded_targets).any(dim=1).float()
    correct_samples = correct_samples.scatter_add_(0, indexes, is_correct)

    per_class_acc = correct_samples / (total_classes_samples_nb + 1e-8)
    classes_samples_nb = (total_classes_samples_nb > 0).float()
    accuracy = (
        (per_class_acc * classes_samples_nb).sum()
        / (classes_samples_nb.sum() + 1e-8)
    )

    if get_acc_per_class:
        return accuracy, correct_samples, total_classes_samples_nb
    return accuracy


def f1(
    outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro'
) -> float:
    """
    Calculates f1-score

    Args:
        predictions:
            Model outputs (logits)
        targets:
            Ground truth labels
        num_classes:
            Number of classes
        average:
            Method to average F1-scores ('macro', 'micro', 'weighted')
    """
    _, predicted = outputs.max(dim=1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    score = f1_score(targets, predicted, average=average, zero_division=0)

    return score
