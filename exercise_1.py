import os
import torch
import shutil
import pandas as pd
from glob import glob

from datetime import datetime

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from galaxy_classification.data import GalaxyZooDataset
from galaxy_classification.networks.cnn import GalaxyClassificationCNN
from galaxy_classification.data import GalaxyZooDataset, get_labels_train

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

#.:O0O:. Functions .:O0O:.

def condition(labels, column_range):
    '''Check if any of the first three label values are greater than or equal to 0.8 (default threshold). 
    If yes, return True.'''
    condition = labels[:, 1:4] >= 0.8
    return torch.sum(condition, dim=1).bool()

def one_hot_encode(labels, threshold=0.8):
    '''Convert labels into one-hot encoded vectors based on a threshold.'''
    one_hot = torch.zeros_like(labels)
    one_hot[labels >= threshold] = 1   # Set to 1 where labels meet the threshold
    return one_hot

# Training function with loss and accuracy tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, output_folder=None):
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct / total)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Evaluate on validation set
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


class GalaxyClassificationCNN(nn.Module):
    def __init__(self, input_image_shape, channel_count_hidden, convolution_kernel_size, mlp_hidden_unit_count, output_size):
        super(GalaxyClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, channel_count_hidden, kernel_size=convolution_kernel_size, stride=1, padding=1)  # Change input channels to 1
        self.conv2 = nn.Conv2d(channel_count_hidden, channel_count_hidden * 2, kernel_size=convolution_kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.fc1 = nn.Linear(channel_count_hidden * 2 * (input_image_shape[0] // 4) * (input_image_shape[1] // 4), mlp_hidden_unit_count)
        self.fc2 = nn.Linear(mlp_hidden_unit_count, output_size)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(torch.relu(self.fc1(x)))  # Apply dropout
        x = self.fc2(x)
        return x    

# Evaluate the model and save predictions
def save_predictions(model, val_loader, output_folder):
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
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

# Save validation labels and optionally move validation images to a separate folder
def save_validation_data(val_dataset, output_folder, validation_images_folder=None):
    # Extract filenames and labels from the validation dataset
    validation_filenames = [os.path.basename(val_dataset.dataset.image_paths[idx]) for idx in val_dataset.indices]
    validation_labels = val_dataset.dataset.labels[val_dataset.indices]

    # Create a DataFrame for validation labels
    validation_df = pd.DataFrame(validation_labels, columns=["Class1.1", "Class1.2", "Class1.3"])
    validation_df.insert(0, "filename", validation_filenames)

    # Save validation labels to a CSV file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    validation_labels_file = os.path.join(output_folder, f"validation_labels_{timestamp}.csv")
    validation_df.to_csv(validation_labels_file, index=False)
    print(f"Validation labels saved to {validation_labels_file}")

    # Optionally move validation images to a separate folder
    if validation_images_folder:
        os.makedirs(validation_images_folder, exist_ok=True)
        for filename in validation_filenames:
            src_path = os.path.join(folder_images_0, filename)
            dst_path = os.path.join(validation_images_folder, filename)
            shutil.copy(src_path, dst_path)
        print(f"Validation images moved to {validation_images_folder}")

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


#.:O0O:. Paths .:O0O:.
folder_0 = "data/exercise_cnnletstryagain/"
folder_images_0 = os.path.join(folder_0, "images/")
file_labels_0 = os.path.join(folder_0, "labels.csv")
labels_file = "data/training_solutions_rev1.csv"
image_folder = "data/images/"

# Ensure output folder exists
if os.path.exists(folder_images_0):
    shutil.rmtree(folder_images_0)
os.makedirs(folder_images_0, exist_ok=True)

# Load labels
labels = get_labels_train(labels_file)  # Labels_file is "data/training_solutions_rev1.csv"
image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))

#.:O0O:. Start Training .:O0O:.

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=20000)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
target_labels = get_labels_train(labels_file)

print("Dataset loaded with", len(dataset), "samples.")

# Loop through and save selected images + labels
files_1 = []
labels_1 = []

for images, labels, paths in data_loader:
    cond = condition(labels)
    selected_indices = torch.where(cond)[0]

    for idx in selected_indices:
        image_tensor = images[idx]
        label_tensor = labels[idx, 1:4]  # First 4 values
        label_tensor = one_hot_encode(label_tensor)  # Convert to one-hot encodin
        path = paths[idx]

        # Save transformed image
        #image_pil = to_pil(image_tensor)
        filename = os.path.basename(path)
        #image_pil.save(os.path.join(folder_images_0, filename))
        save_image(image_tensor, os.path.join(folder_images_0, filename))

        # Save label
        labels_1.append(label_tensor)  # Optional rounding
        files_1.append(filename)

# Save filtered labels
labels_1 = torch.stack(labels_1)
labels_df = pd.DataFrame(labels_1.numpy(), columns=["Type1.1", "Type1.2", "Type1.3"])
#labels_df = pd.DataFrame(labels_1.numpy(), columns=target_labels.columns[:4])
labels_df.insert(0, 'filename', files_1)
labels_df.to_csv(file_labels_0, index=False)

# Extract image paths and labels
image_paths = [os.path.join(folder_images_0, filename) for filename in labels_df["filename"]]
labels = labels_df[["Type1.1", "Type1.2", "Type1.3"]].values  # Columns 2, 3, and 4

# Create dataset
galaxy_dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels)

# Split dataset into training and validation
train_size = int(0.8 * len(galaxy_dataset))
val_size = len(galaxy_dataset) - train_size
train_dataset, val_dataset = random_split(galaxy_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the CNN model with dropout for regularization
input_image_shape = (64, 64)  # Input images are 64x64
channel_count_hidden = 32
convolution_kernel_size = 3
mlp_hidden_unit_count = 128
output_size = 3  # Three output classes Class1.1, Class1.2, Class1.3)

model = GalaxyClassificationCNN(
    input_image_shape=input_image_shape,
    channel_count_hidden=channel_count_hidden,
    convolution_kernel_size=convolution_kernel_size,
    mlp_hidden_unit_count=mlp_hidden_unit_count,
    output_size=output_size,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)  # Reduced learning rate

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, output_folder=folder_0)

# Save predictions
save_predictions(model, val_loader, folder_0)

# Save validation data
validation_images_folder = os.path.join(folder_0, "validation_images")
save_validation_data(val_dataset, folder_0, validation_images_folder)

# Evaluate and save diagnostics
evaluate_and_save_diagnostics(model, val_loader, folder_0)