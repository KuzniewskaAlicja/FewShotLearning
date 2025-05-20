from typing import Any
from collections.abc import Callable
import random
from functools import partial
from pathlib import Path

import torch

from src.dataset.transforms import mixup_data
from src.evaluation import validate, mean_per_class_accuracy, accuracy, f1


random.seed(42)


class ModelTrainer:
    def __init__(self, model: Any, device: torch.device, classes_nb: int):
        self.model = model
        self.device = device
        self.classes_nb = classes_nb
        self.f1_score_op = partial(f1, average="macro")
        self.class_acc_op = partial(
            mean_per_class_accuracy,
            num_classes=self.classes_nb,
            get_acc_per_class=False
        )

    def configure_stage(self, stage: str, epoch_num: int, epoch_steps: int):
        """
        Configures the training setup for a specified training stage

        Args:
            stage:
                Name of the training stage
            epoch_num:
                Total number of epochs for the current training stage
            epoch_steps:
                Number of steps (batches) per epoch
        """
        self.stage = stage
        self.epoch_num = epoch_num

        if self.stage == "head_only":
            self.model.freeze_backbone()
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=1e-3,
                weight_decay=0.01
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=2e-3,
                epochs=self.epoch_num,
                steps_per_epoch=epoch_steps,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=10000
            )
        elif self.stage == "partial_unfreezing":
            self.model.unfreeze_model("last_stage")
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        'params': filter(
                            lambda p: p.requires_grad,
                            self.model.backbone.parameters()
                        ),
                        'lr': 1e-5,
                        "weight_decay": 0.02
                    },
                    {
                        'params': self.model.head.parameters(),
                        'lr': 5e-4,
                        "weight_decay": 0.01
                    }
                ]
            )
            self.scheduler = \
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
                )
        elif stage == "full_model":
            self.model.unfreeze_model("full")
            self.optimizer = torch.optim.AdamW(
                {'params': self.model.backbone.parameters(), 'lr': 8e-6},
                {'params': self.model.head.parameters(), 'lr': 1e-4}
            )
            self.scheduler = \
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
                )

    def run_stage(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: Any,
        patience: int,
        save_path: Path
    ):
        """
        Executes training stage of a model

        Args:
            train_lader:
                Pytorch DataLoader of training data
            val_loader:
                Pytorch DataLoader of validation data
            criterion:
                Loss function
            patience:
                Number of epochs with no improvement
                after which the training will stop
            save_path:
                Path where best models checkpoints will be saved
        """
        print(10 * "=", f"Phase: {self.stage}", 10*'=')
        self.criterion = criterion
        best_f1 = 0.0
        best_weights = None
        no_improve = 0

        for epoch in range(self.epoch_num):
            print(20 * '-' + f"Epoch {epoch+1}/{self.epoch_num}" + 20 * '-')

            train_metrics = self._train_epoch(
                train_loader=train_loader, mixup_alpha=0.2, mixup_prob=0.5
            )

            val_metrics = validate(
                model=self.model,
                val_loader=val_loader,
                criterion=self.criterion,
                device=self.device,
                get_acc_per_class=False
            )

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                "\t\tTrain results\n"
                f"Loss: {train_metrics['loss']:.4f}\n"
                f"Acc: {train_metrics['accuracy']:.2f}\n"
                f"Mean acc per class {train_metrics['mean_acc_per_class']:.2f}\n"
                f"F1 (macro): {train_metrics['f1_macro']:.4f}\n"
            )
            print(
                "\t\tVal results\n"
                f"Loss: {val_metrics['loss']:.4f}\n"
                f"Acc: {val_metrics['accuracy']:.2f}\n"
                f"Mean acc per class {val_metrics['mean_acc_per_class']:.2f}\n"
                f"F1 (macro): {val_metrics['f1_macro']:.4f}\n"
                f"F1 (micro): {val_metrics['f1_micro']:.4f}\n"
                f"F1 (weighted): {val_metrics['f1_weighted']:.4f}\n"
            )

            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_weights = self.model.state_dict()
                torch.save(
                    self.model, save_path / f'best_{self.stage}_model.pt'
                )
                print(f"✅ Model saved with best f1: {best_f1:.4f}")
                no_improve = 0
            else:
                no_improve += 1

                if no_improve >= patience:
                    print(
                        f"🛑 Early stopping after {patience} "
                        "epochs without improvement"
                    )
                    break
        self.model.load_state_dict(best_weights)

    def save_final_model(self, save_path: Path):
        torch.save(self.model, save_path / "final_model.pt")

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5
    ) -> dict[str, float]:
        """
        Trains the model for one epoch using the provided data loader

        Args:
            train_loader:
                Pytorch loader of training data
            mixup_alpha:
                Alpha parameter for the Beta distribution used in MixUp
            mixup_prob:
                Probability of applying MixUp to a given batch
        """
        self.model.train()
        running_loss = 0.0
        acc, acc_per_class, f1_value = 0.0, 0.0, 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if mixup_prob and random.random() < mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, device=self.device, alpha=mixup_alpha
                )

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.adjust_to_mixup(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
                acc += self.adjust_to_mixup(
                    accuracy, outputs, targets_a, targets_b, lam
                )
                acc_per_class += self.adjust_to_mixup(
                    self.class_acc_op, outputs, targets_a, targets_b, lam
                )
                f1_value += self.adjust_to_mixup(
                    self.f1_score_op, outputs, targets_a, targets_b, lam
                )
            else:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                acc += accuracy(outputs, targets)
                acc_per_class += self.class_acc_op(outputs, targets)
                f1_value += self.f1_score_op(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = f1_value / len(train_loader)
        epoch_acc = acc / len(train_loader)
        epoch_acc_per_class = acc_per_class / len(train_loader)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "mean_acc_per_class": epoch_acc_per_class,
            "f1_macro": epoch_f1,
        }

    @staticmethod
    def adjust_to_mixup(
        operation: Callable,
        predictions: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        lam: float
    ) -> float:
        """
        Adjusts provided operation for MixUp-augmented inputs

        Args:
            operation:
                Operation to execute
            predictions:
                Model outputs (logits)
            labels_a:
                First set of ground truth labels (mixed input A)
            labels_b:
                Second set of ground truth labels (mixed input B)
            lam:
                MixUp coefficient representing the weight of labels_a
        """
        return (
            lam * operation(predictions, labels_a)
            + (1 - lam) * operation(predictions, labels_b)
        )
