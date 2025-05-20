import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from src.models import ConvNeXtModel
from src.training import ModelTrainer
from src.dataset import load_data, get_dataset_info


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=Path,
    required=True,
    help="Path to the training dataset"
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Number of epoch for waiting for model improvement"
)
parser.add_argument(
    "--save_path",
    type=Path,
    default=Path("./models"),
    help="Directory where model will be saved"
)

args = parser.parse_args()


def train_with_progressive_unfreezing(
    trainer: ModelTrainer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader
):
    """
    Runs training with progressive model unfreezing

    Args:
        trainer:
            Model trainer
        train_loader:
            Pytorch dataloader for training data
        val_loader:
            Pytorch dataloader for validation data
    """
    criterion = torch.nn.CrossEntropyLoss()
    trainer.configure_stage(
        stage="head_only", epoch_num=25, epoch_steps=len(train_loader)
    )
    trainer.run_stage(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        patience=args.patience,
        save_path=args.save_path
    )

    trainer.configure_stage(
        stage="partial_unfreezing",
        epoch_num=40,
        epoch_steps=len(train_loader)
    )
    trainer.run_stage(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        patience=args.patience,
        save_path=args.save_path
    )
    trainer.save_final_model(args.save_path)


if __name__ == "__main__":
    image_paths, labels, classes = get_dataset_info(args.dataset_path)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    train_loader = load_data(
        train_paths,
        train_labels,
        classes,
        img_size=224,
        dataset_type="train",
        batch_size=64
    )
    val_loader = load_data(
        val_paths,
        val_labels,
        classes,
        img_size=224,
        dataset_type="val",
        batch_size=128
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtModel(class_nb=len(classes)).to(device)
    trainer = ModelTrainer(model, device, len(classes))

    if not args.save_path.exists():
        args.save_path.mkdir(parents=True, exist_ok=True)

    train_with_progressive_unfreezing(trainer, train_loader, val_loader)
