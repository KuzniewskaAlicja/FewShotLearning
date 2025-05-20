import argparse
from pathlib import Path
import torch

from src.evaluation import (
    validate, create_results_description, plot_predictions
)
from src.dataset import load_data, get_dataset_info


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=Path,
    required=True,
    help="Path to the validation dataset"
)
parser.add_argument(
    "--model_path",
    type=Path,
    required=True,
    help="Path to the trained model"
)
parser.add_argument(
    "--results_dir",
    type=Path,
    default="./results",
    help="Directory path where the result of a model will be saved"
)
args = parser.parse_args()

if __name__ == "__main__":
    data_paths, labels, classes = get_dataset_info(args.dataset_path)
    loader = load_data(
        data_paths,
        labels,
        classes,
        img_size=224,
        dataset_type="val",
        batch_size=64
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(
        args.model_path, weights_only=False, map_location=device
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(10 * '=', "Model evaluation", 10 * '=')
    top1_metrics = validate(
        model, loader, criterion, device, topk=1, get_acc_per_class=True
    )
    top5_metrics = validate(
        model, loader, criterion, device, topk=5, get_acc_per_class=True
    )

    print(
        f"Accuracy: Top-1 - {top1_metrics['accuracy']:.2f}, "
        f"Top-5 - {top5_metrics['accuracy']:.2f}\n"
        f"F1 (macro): {top1_metrics['f1_macro']:.2f}"
    )

    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True, exist_ok=True)

    with (args.results_dir / "metrics_results.txt").open("w") as file:
        file.write(
            create_results_description(top1_metrics, top5_metrics, classes)
        )

    fig = plot_predictions(model, loader, 15, device)
    fig.savefig(args.results_dir / "example_predictions.png")

    print(40 * '=', f"\nResult saved in {args.results_dir.absolute()}")
