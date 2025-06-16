from collections.abc import Callable
from matplotlib import pyplot
from pathlib import Path
from torch import Tensor
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingSummary:
    printing_interval_epochs: int
    epoch_index: int
    training_losses: list[float]
    training_accuracies: list[float]
    validation_losses: list[float]
    validation_accuracies: list[float]

    def __init__(self, printing_interval_epochs: int):
        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.printing_interval_epochs = printing_interval_epochs
        self.epoch_index = 0

    def append_epoch_summary(
        self,
        training_loss: float,
        training_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
    ):
        if (self.epoch_index + 1) % self.printing_interval_epochs == 0:
            print(
                f"epoch {self.epoch_index + 1}, training loss {training_loss:.2e}, training accuracy {training_accuracy * 100.0:.2f}%"
                f" validation loss {validation_loss:.2e}, validation accuracy {validation_accuracy * 100.0:.2f}%"
            )

        self.training_losses.append(training_loss)
        self.training_accuracies.append(training_accuracy)
        self.validation_losses.append(validation_loss)
        self.validation_accuracies.append(validation_accuracy)
        self.epoch_index += 1

    def save_plot(self, path: Path):
        figure, axes_loss = pyplot.subplots()

        epoch_numbers = list(range(self.epoch_index))
        axes_loss.plot(
            epoch_numbers, self.training_losses, label="training loss", color="C0"
        )
        axes_loss.plot(
            epoch_numbers, self.validation_losses, label="validation loss", color="C1"
        )
        axes_loss.set_xlabel("epoch")
        axes_loss.set_ylabel("loss")
        axes_loss.legend()

        axes_accuracy = axes_loss.twinx()
        axes_accuracy.plot(
            epoch_numbers,
            self.training_accuracies,
            label="training accuracy",
            color="C0",
            linestyle="dashed",
        )
        axes_accuracy.plot(
            epoch_numbers,
            self.validation_accuracies,
            label="validation accuracy",
            color="C1",
            linestyle="dashed",
        )
        axes_accuracy.set_ylabel("accuracy")
        axes_accuracy.legend()

        figure.savefig(path, bbox_inches="tight")


def compute_average_epoch_loss(
    model: Module,
    dataloader: DataLoader,
    loss: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer | None = None,
):
    epoch_loss_train = 0.0
    for batch in dataloader:
        images, labels = batch["images"], batch["labels"]

        if optimizer is not None:
            optimizer.zero_grad()

        labels_predicted = model(images)
        loss_batch = loss(labels_predicted, labels)

        if optimizer is not None:
            loss_batch.backward()
            optimizer.step()

        epoch_loss_train += loss_batch.item()

    return epoch_loss_train / len(dataloader)


def compute_accuracy(model: Module, dataloader: DataLoader) -> float:
    # the prediction_count is not len(dataloader) * dataloader.batch_size,
    prediction_count = 0
    correct_prediction_count = 0
    for batch in dataloader:
        images, labels = batch["images"], batch["labels"]
        labels_predicted = model(images)

        labels_predicted = labels_predicted.argmax(dim=1)
        correct_prediction_count += torch.sum(
            labels_predicted == labels
        ).item()
        prediction_count += len(batch["images"])

    return correct_prediction_count / prediction_count


def fit(
    network: Module,
    optimizer: Optimizer,
    loss: Callable[[Tensor, Tensor], Tensor],
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epoch_count: int,
) -> TrainingSummary:
    summary = TrainingSummary(printing_interval_epochs=1)
    for _ in range(epoch_count):
        network.train()
        epoch_loss_training = compute_average_epoch_loss(
            network, training_dataloader, loss, optimizer
        )
        epoch_accuracy_training = compute_accuracy(network, training_dataloader)

        network.eval()
        epoch_loss_validation = compute_average_epoch_loss(
            network, validation_dataloader, loss
        )
        epoch_accuracy_validation = compute_accuracy(network, validation_dataloader)

        summary.append_epoch_summary(
            epoch_loss_training,
            epoch_accuracy_training,
            epoch_loss_validation,
            epoch_accuracy_validation,
        )

    return summary


def evaluate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels, _ in val_loader:  # Ignore the third value (paths)
            images, labels = images.to(device), labels.to(device)
            targets = labels.argmax(dim=1)  # Convert one-hot labels to class indices
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class indices
            correct += (preds == targets).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_model_with_accuracy(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    accuracy = compute_accuracy(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy
