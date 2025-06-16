import logging
import os
import shutil
import simple_parsing
import yaml
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from pathlib import Path
import matplotlib.pyplot as plt

from dataclasses import asdict, dataclass
from galaxy_classification.data import (
    SplitDataLoader,
    GalaxyZooDataset,
)
from galaxy_classification.networks import (
    NetworkConfig, 
    build_network, 
    GalaxyClassificationCNN
)

from evaluate import evaluate_model

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingCli:
    run_name: str
    no_config_edit: bool = False

@dataclass
class TrainingConfig:
    epoch_count: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    network: NetworkConfig

def load_config(path: Path) -> TrainingConfig:
    with open(path) as config_file:
        return dacite.from_dict(TrainingConfig, yaml.safe_load(config_file)["training"])

def prepare_config(
    output_path: Path, default_path: Path, run_name: str, allow_config_edit: bool
) -> TrainingConfig:
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"copying {default_path} to {output_path}")
    shutil.copy(default_path, output_path)
    if allow_config_edit:
        _ = input(
            f"please edit the config in outputs/{run_name}/config.yaml"
            " to set the parameters for this run\n"
            "afterwards, please press enter to continue..."
        )
    return load_config(output_path)

def save_hyperparameters(path: Path, config: NetworkConfig):
    with open(path, "w") as hyperparameter_cache:
        yaml.dump(asdict(config), hyperparameter_cache)

# Training function with loss and accuracy tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output_folder=None):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device).argmax(dim=1)  # Convert one-hot labels to class indices
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Convert logits to predicted class indices
            _, predicted = torch.max(outputs, 1)

            # Convert one-hot encoded labels to class indices if necessary
            if labels.ndim == 2:  # Check if labels are one-hot encoded
                labels = torch.argmax(labels, dim=1)

            # Compare predicted class indices with ground truth
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct / total)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Evaluate on validation set
        # included here due to issues when included in evaluate.py
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot loss and accuracy curves
    if output_folder:
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, output_folder)

# Function to plot training curves
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, output_folder):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    loss_curve_file = os.path.join(output_folder, "loss_curve.png")
    plt.savefig(loss_curve_file)
    print(f"Loss curve saved to {loss_curve_file}")

    # Plot accuracy curve
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color="blue")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    accuracy_curve_file = os.path.join(output_folder, "accuracy_curve.png")
    plt.savefig(accuracy_curve_file)
    print(f"Accuracy curve saved to {accuracy_curve_file}")

def main(galaxy_dataset, output_folder):
    cli = simple_parsing.parse(TrainingCli)
    config = prepare_config(
        Path(f"outputs/{cli.run_name}/config.yaml"),
        Path("config_default.yaml"),
        cli.run_name,
        not cli.no_config_edit,
    )

    print("loading images and labels for Galaxy Zoo dataset")

    # Split dataset into training and validation
    galaxy_split_loader = SplitDataLoader(
        dataset_prepared=galaxy_dataset,
        validation_fraction=config.validation_fraction,
        batch_size=config.batch_size,
    )

    # Access training and validation DataLoaders
    train_loader = galaxy_split_loader.training_dataloader
    val_loader = galaxy_split_loader.validation_dataloader

    # Extract parameters from the config
    input_image_shape = tuple(config.network.input_image_shape)
    channel_count_hidden = config.network.channel_count_hidden
    convolution_kernel_size = config.network.convolution_kernel_size
    mlp_hidden_unit_count = config.network.mlp_hidden_unit_count
    output_size = config.network.output_size

    # Initialize the model using parameters from the config
    model = GalaxyClassificationCNN(
        input_image_shape=input_image_shape,
        channel_count_hidden=channel_count_hidden,
        convolution_kernel_size=convolution_kernel_size,
        mlp_hidden_unit_count=mlp_hidden_unit_count,
        output_size=output_size,
    )

    model.to(device)

    # Save hyperparameters to a YAML file
    #hyperparameters_path = Path(output_folder) / "evaluation" / "hyperparameters.yaml"
    #hyperparameters_path.parent.mkdir(parents=True, exist_ok=True)
    #save_hyperparameters(hyperparameters_path, config.network)
    #print(f"Hyperparameters saved to {hyperparameters_path}")

    # Define loss function and optimizer
    ###criterion = CrossEntropyLoss()
    #optimizer = AdamW(model.parameters(), lr=0.0001) 
    ###optimizer = AdamW(model.parameters(), config.learning_rate) 
 
    # wanted to use this, however cannot solve an error message
    print(f"training the {config.network.network_id} classifier")
    network = build_network(
        input_image_shape,
        config.network,
    )
    optimizer = AdamW(network.parameters(), lr=config.learning_rate)
    loss = CrossEntropyLoss()
    #training_summary = GalaxyClassificationCNN(
    #    network,
    #    optimizer,
    #    loss,
    #    galaxy_split_dataloader.training_dataloader,
    #    galaxy_split_dataloader.validation_dataloader,
    #    config.epoch_count,
    #)
    # Train the model
    training_summary = train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        num_epochs=config.epoch_count,
        output_folder=output_folder,
    )

    return val_loader, model #use validation set later for evaluation


if __name__ == "__main__":
    logging.basicConfig()
    main()