import os
import torch
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from pathlib import Path
import yaml
import dacite
from galaxy_classification.networks import build_network, GalaxyClassificationCNN
from galaxy_classification.data import GalaxyZooDataset
from dataclasses import dataclass
from typing import get_args
import simple_parsing
from torch.utils.data import DataLoader

# Define device globally
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EvaluationCli:
    run_name: str

@dataclass
class EvaluationConfig:
    network: dict  # Network configuration
    batch_size: int

def load_config(path: Path) -> EvaluationConfig:
    with open(path) as config_file:
        return dacite.from_dict(EvaluationConfig, yaml.safe_load(config_file)["training"])

# Evaluation function with loss calculation
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(device), labels.to(device).argmax(dim=1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Save validation labels and optionally move validation images to a separate folder
def save_validation_data(val_dataset, output_folder, validation_images_folder=None):
    # Extract filenames and labels from the validation dataset
    validation_filenames = [os.path.basename(path) for path in val_dataset.image_paths]
    validation_labels = val_dataset.labels

    # Create a DataFrame for validation labels
    validation_df = pd.DataFrame(validation_labels, columns=["Class1.1", "Class1.2", "Class1.3"])
    validation_df.insert(0, "filename", validation_filenames)

    # Save validation labels to a CSV file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    validation_labels_file = os.path.join(output_folder, f"validation_labels_{timestamp}.csv")
    validation_df.to_csv(validation_labels_file, index=False)
    print(f"Validation labels saved to {validation_labels_file}")

def save_predictions(model, val_loader, output_folder):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)  # Use the globally defined device
            outputs = model(images)
            predicted = torch.softmax(outputs, dim=1).cpu().numpy()  # Get probabilities for each class
            predictions.extend(predicted)
            filenames.extend([os.path.basename(path) for path in paths])  # Extract filenames

    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame(predictions, columns=["Predicted Class1.1", "Predicted Class1.2", "Predicted Class1.3"])
    predictions_df.insert(0, "filename", filenames)

    # Save predictions to a CSV file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    output_file = os.path.join(output_folder, f"predictions_{timestamp}.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Function to count actual and predicted galaxies for each type
def count_galaxies(labels, predictions, output_folder):
    # Ensure labels and predictions are NumPy arrays
    labels = labels.numpy()
    predictions = predictions.numpy()

    # Convert predictions to binary (threshold at 0.5)
    binary_predictions = (predictions >= 0.5).astype(int)

    # Count actual galaxies for each type
    actual_counts = labels.sum(axis=0)
    predicted_counts = binary_predictions.sum(axis=0)

    # Print counts for each class
    print("\nGalaxy Counts:")
    for i, class_name in enumerate(["Class1.1", "Class1.2", "Class1.3"]):
        print(f"{class_name}: Actual Count = {actual_counts[i]}, Predicted Count = {predicted_counts[i]}")

    # Create a DataFrame for counts
    counts_df = pd.DataFrame({
        "Type": ["Class1.1", "Class1.2", "Class1.3"],
        "Actual Count": actual_counts,
        "Predicted Count": predicted_counts
    })

    # Save counts to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counts_file = os.path.join(output_folder, f"galaxy_counts_{timestamp}.csv")
    counts_df.to_csv(counts_file, index=False)
    print(f"Galaxy counts saved to {counts_file}")


# Function to plot and save ROC curves
def plot_roc_curve(labels, predictions, output_folder):
    # Ensure labels and predictions are NumPy arrays
    labels = labels.numpy()
    predictions = predictions.numpy()

    # Create a ROC curve for each class
    for i, class_name in enumerate(["Class1.1", "Class1.2", "Class1.3"]):
        fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {class_name}")
        plt.legend(loc="lower right")

        # Save the ROC curve to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        roc_file = os.path.join(output_folder, f"roc_curve_{class_name}_{timestamp}.png")
        plt.savefig(roc_file)
        print(f"ROC curve for {class_name} saved to {roc_file}")   
      

# Function to evaluate model performance per class
def evaluate_class_performance(labels, predictions, output_folder):
    # Ensure labels and predictions are NumPy arrays
    labels = labels.numpy()
    predictions = predictions.numpy()

    # Convert predictions to binary (threshold at 0.5)
    binary_predictions = (predictions >= 0.5).astype(int)

    # Generate classification report for each class
    report = classification_report(
        labels,
        binary_predictions,
        target_names=["Class1.1", "Class1.2", "Class1.3"],
        output_dict=True,
    )

    # Save classification report to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_folder, f"classification_report_{timestamp}.csv")
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_file, index=True)
    print(f"Classification report saved to {report_file}")

    # Identify best and worst performing classes
    f1_scores = report_df["f1-score"][:-3]  # Exclude averages (last 3 rows)
    best_class = f1_scores.idxmax()
    worst_class = f1_scores.idxmin()
    print(f"Best performing class: {best_class} (F1-score: {f1_scores[best_class]:.2f})")
    print(f"Worst performing class: {worst_class} (F1-score: {f1_scores[worst_class]:.2f})")


# Evaluate the model and save diagnostics
def evaluate_and_save_diagnostics(model, val_loader, output_folder):
    model.eval()
    predictions = []
    labels = []
    filenames = []
    with torch.no_grad():
        for images, batch_labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = torch.softmax(outputs, dim=1).cpu()  # Get probabilities for each class
            predictions.append(predicted)
            labels.append(batch_labels.cpu())
            filenames.extend([os.path.basename(path) for path in paths])  # Extract filenames

    # Stack predictions and labels
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions.numpy(), columns=["Predicted Class1.1", "Predicted Class1.2", "Predicted Class1.3"])
    predictions_df.insert(0, "filename", filenames)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"predictions_{timestamp}.csv")
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Plot and save ROC curves
    plot_roc_curve(labels, predictions, output_folder)

    # Evaluate class performance
    evaluate_class_performance(labels, predictions, output_folder)

    # Count galaxies for each type
    count_galaxies(labels, predictions, output_folder)         

def main(val_loader, galaxy_dataset, model, output_folder):
    # Call the function to save predictions
    save_predictions(model, val_loader, output_folder)

    # Call the function to save validation data
    validation_images_folder = os.path.join(output_folder, "validation_images")
    save_validation_data(galaxy_dataset, output_folder, validation_images_folder)
    
    # Call the function to evaluate and save diagnostics
    evaluate_and_save_diagnostics(model, val_loader, output_folder)
 
    
if __name__ == "__main__":
    main()
