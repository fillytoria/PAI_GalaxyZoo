import os
import torch
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms 

from digit_classification.data import GalaxyZooDataset, get_labels_train

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import datetime # Timestamp for saving files
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#.:O0O:. Paths .:O0O:.
folder_3 = "data/exercise_3/"
folder_images_3 = os.path.join(folder_3, "images/")
file_labels_3 = os.path.join(folder_3, "labels.csv")
folder_evaluation_3 = os.path.join(folder_3, "evaluation/")
os.makedirs(folder_images_3, exist_ok=True)
os.makedirs(folder_evaluation_3, exist_ok=True)

# Create a unique folder for this run using the timestamp
run_folder = os.path.join(folder_evaluation_3, f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# Load labels and images
labels = get_labels_train("data/training_solutions_rev1.csv")
image_paths = sorted(glob("data/images/*.jpg"))

#.:O0O:. Extract relevant columns for regression.:O0O:.

# Extract relevant columns for Q2, Q7, Q6, and Q8
q2_columns = [5, 6]  # Q2: [Class2.1, Class2.2]
q7_columns = [17, 18, 19]  # Q7: [Class7.1, Class7.2, Class7.3]
q6_columns = [15, 16]  # Q6: [Class6.1, Class6.2]
q8_columns = [21, 22, 23, 24, 25, 26, 27]  # Q8: [Class8.1 to Class8.7]
relevant_columns = q2_columns + q7_columns + q6_columns + q8_columns
labels = labels[:, relevant_columns]  # Extract only the relevant columns

# Save relevant labels to a new CSV file
relevant_labels_df = pd.DataFrame(
    labels.numpy(),
    columns=[
        "Class2.1", "Class2.2",
        "Class7.1", "Class7.2", "Class7.3",
        "Class6.1", "Class6.2",
        "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6", "Class8.7"
    ]
)
relevant_labels_df.to_csv(file_labels_3, index=False)
print(f"Relevant labels saved to {file_labels_3}")

# Verify the number of images and labels
print(f"Number of images: {len(image_paths)}")
print(f"Number of labels: {len(labels)}")

# Ensure consistent slicing
max_samples = 500
image_paths = image_paths[:max_samples]
labels = labels[:max_samples]

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=max_samples)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the model
class ImprovedRegressionModel(nn.Module):
    def __init__(self):
        super(ImprovedRegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 256)
        self.fc2 = nn.Linear(256, 14)  # Output size matches the number of regression targets

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = ImprovedRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluate the model
model.eval()
all_predictions = []
all_ground_truth = []
with torch.no_grad():
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        all_predictions.append(outputs.cpu().numpy())
        all_ground_truth.append(labels.cpu().numpy())

# Convert lists to NumPy arrays
all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truth = np.concatenate(all_ground_truth, axis=0)

# Calculate metrics
mse = mean_squared_error(all_ground_truth, all_predictions)
mae = mean_absolute_error(all_ground_truth, all_predictions)
r2 = r2_score(all_ground_truth, all_predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save metrics for non-augmented training
metrics_file = os.path.join(run_folder, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
print(f"Metrics saved to {metrics_file}")

# Save plots for non-augmented training
target_names = [
    "Class2.1", "Class2.2",
    "Class7.1", "Class7.2", "Class7.3",
    "Class6.1", "Class6.2",
    "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6", "Class8.7"
]
for i, target_name in enumerate(target_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(all_ground_truth[:, i], all_predictions[:, i], alpha=0.7, label=target_name)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title(f"Predictions vs. Ground Truth for {target_name}")
    plt.legend()
    plot_file = os.path.join(run_folder, f"predictions_vs_ground_truth_{target_name}.png")
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")
    plt.close()

# Define the first model for Q2 and Q7
class Q2Q7Model(nn.Module):
    def __init__(self):
        super(Q2Q7Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 5)  # Output size for Q2 and Q7

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

q2q7_model = Q2Q7Model().to(device)
# Train q2q7_model as usual


#Use predictions as features for Q6 and Q8
# Define the second model for Q6 and Q8
class Q6Q8Model(nn.Module):
    def __init__(self):
        super(Q6Q8Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64 + 5, 256)  # Include 5 predictions from Q2 and Q7
        self.fc2 = nn.Linear(256, 9)  # Output size for Q6 and Q8

    def forward(self, x, q2q7_predictions):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.cat([x, q2q7_predictions], dim=1)  # Concatenate predictions
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

q6q8_model = Q6Q8Model().to(device)


# Train the Q2Q7 model
print("\nTraining Q2Q7 Model...")
q2q7_optimizer = torch.optim.Adam(q2q7_model.parameters(), lr=0.001)
q2q7_criterion = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    q2q7_model.train()
    epoch_loss = 0
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels[:, :5].to(device)  # Only Q2 and Q7 labels
        q2q7_optimizer.zero_grad()
        outputs = q2q7_model(images)
        loss = q2q7_criterion(outputs, labels)
        loss.backward()
        q2q7_optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Q2Q7 Model, Loss: {epoch_loss:.4f}")

# Evaluate the Q2Q7 model
q2q7_model.eval()
q2q7_predictions = []
q2q7_ground_truth = []
with torch.no_grad():
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels[:, :5].to(device)  # Only Q2 and Q7 labels
        outputs = q2q7_model(images)
        q2q7_predictions.append(outputs.cpu().numpy())
        q2q7_ground_truth.append(labels.cpu().numpy())

q2q7_predictions = np.concatenate(q2q7_predictions, axis=0)
q2q7_ground_truth = np.concatenate(q2q7_ground_truth, axis=0)

# Train the Q6Q8 model using Q2Q7 predictions as input
print("\nTraining Q6Q8 Model...")
q6q8_optimizer = torch.optim.Adam(q6q8_model.parameters(), lr=0.001)
q6q8_criterion = nn.MSELoss()

for epoch in range(num_epochs):
    q6q8_model.train()
    epoch_loss = 0
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels[:, 5:].to(device)  # Only Q6 and Q8 labels
        with torch.no_grad():
            q2q7_predictions = q2q7_model(images)  # Get Q2/Q7 predictions
        q6q8_optimizer.zero_grad()
        outputs = q6q8_model(images, q2q7_predictions)
        loss = q6q8_criterion(outputs, labels)
        loss.backward()
        q6q8_optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Q6Q8 Model, Loss: {epoch_loss:.4f}")

# Evaluate the Q6Q8 model
q6q8_model.eval()
q6q8_predictions = []
q6q8_ground_truth = []
with torch.no_grad():
    for images, labels, _ in data_loader:
        images, labels = images.to(device), labels[:, 5:].to(device)  # Only Q6 and Q8 labels
        q2q7_predictions = q2q7_model(images)  # Get Q2/Q7 predictions
        outputs = q6q8_model(images, q2q7_predictions)
        q6q8_predictions.append(outputs.cpu().numpy())
        q6q8_ground_truth.append(labels.cpu().numpy())

q6q8_predictions = np.concatenate(q6q8_predictions, axis=0)
q6q8_ground_truth = np.concatenate(q6q8_ground_truth, axis=0)

# Calculate metrics for Q6Q8 model
q6q8_mse = mean_squared_error(q6q8_ground_truth, q6q8_predictions)
q6q8_mae = mean_absolute_error(q6q8_ground_truth, q6q8_predictions)
q6q8_r2 = r2_score(q6q8_ground_truth, q6q8_predictions)
print(f"Q6Q8 Model - Mean Squared Error (MSE): {q6q8_mse:.4f}")
print(f"Q6Q8 Model - Mean Absolute Error (MAE): {q6q8_mae:.4f}")
print(f"Q6Q8 Model - R² Score: {q6q8_r2:.4f}")

# Save Q6Q8 metrics
q6q8_metrics_file = os.path.join(run_folder, "q6q8_metrics.txt")
with open(q6q8_metrics_file, "w") as f:
    f.write(f"Q6Q8 Model - Mean Squared Error (MSE): {q6q8_mse:.4f}\n")
    f.write(f"Q6Q8 Model - Mean Absolute Error (MAE): {q6q8_mae:.4f}\n")
    f.write(f"Q6Q8 Model - R² Score: {q6q8_r2:.4f}\n")
print(f"Q6Q8 metrics saved to {q6q8_metrics_file}")

# Save plots for Q6Q8 model
for i, target_name in enumerate(target_names[5:]):  # Only Q6 and Q8 targets
    plt.figure(figsize=(8, 6))
    plt.scatter(q6q8_ground_truth[:, i], q6q8_predictions[:, i], alpha=0.7, label=target_name)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title(f"Q6Q8 Predictions vs. Ground Truth for {target_name}")
    plt.legend()
    q6q8_plot_file = os.path.join(run_folder, f"q6q8_predictions_vs_ground_truth_{target_name}.png")
    plt.savefig(q6q8_plot_file)
    print(f"Q6Q8 plot saved to {q6q8_plot_file}")
    plt.close()