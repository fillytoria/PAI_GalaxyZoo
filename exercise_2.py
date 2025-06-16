import os
import torch
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn 
from torch.utils.data import DataLoader

from galaxy_classification import evaluate_model
from galaxy_classification.networks.cnn import GalaxyClassificationCNN
from galaxy_classification.data import GalaxyZooDataset, get_labels_train

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import datetime # Save plots with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#.:O0O:. Paths .:O0O:.
folder_2 = "data/exercise_2/"
file_labels_2 = os.path.join(folder_2, "labels.csv")
folder_evaluation_2 = os.path.join(folder_2, "evaluation/")

os.makedirs(folder_images_2, exist_ok=True)

# Load labels and images
labels = get_labels_train("data/training_solutions_rev1.csv")  # Load all labels
image_paths = sorted(glob("data/images/*.jpg"))

#.:O0O:. Extract relevant columns for regression.:O0O:.

# For Q2: [Class2.1, Class2.2], For Q7: [Class7.1, Class7.2, Class7.3]
q2_columns = [5, 6]  
q7_columns = [17, 18, 19]  
relevant_columns = q2_columns + q7_columns
labels = labels[:, relevant_columns]  # Extract only relevant columns

# Save relevant labels to a new CSV file
relevant_labels_df = pd.DataFrame(
    labels.numpy(),  # Convert tensor to NumPy array
    columns=["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]  # Column names
)
relevant_labels_df.to_csv(file_labels_2, index=False)
print(f"Relevant labels saved to {file_labels_2}")

#.:O0O:. Start Training .:O0O:.

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=500)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Perform regression
for images, labels, paths in data_loader:
    # Verify constraints
    q2_labels = labels[:, :2]  # First two columns are Q2 (Class2.1, Class2.2)
    q7_labels = labels[:, 2:]  # Last three columns are Q7 (Class7.1, Class7.2, Class7.3)

    # Normalize Q2 and Q7 labels
    #q2_labels = q2_labels / torch.sum(q2_labels, dim=1, keepdim=True)
    #q7_labels = q7_labels / torch.sum(q7_labels, dim=1, keepdim=True)

    # Verify the sums again
    q2_sums = torch.sum(q2_labels, dim=1)
    q7_sums = torch.sum(q7_labels, dim=1)
    #print("Normalized Q2 sums (should be 1):", q2_sums)
    #print("Normalized Q7 sums (should be 1):", q7_sums)

# Initialize the model
model = GalaxyClassificationCNN(
    input_image_shape=(64, 64),  # 64x64 input images
    channel_count_hidden=16,  # Number of hidden channels
    convolution_kernel_size=3,  # Kernel size for convolutions
    mlp_hidden_unit_count=128,  # Number of hidden units in the MLP
    output_size=5,  # Output size for regression (5 values: 2xQ2 + 3xQ7)
)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, labels, _ in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    for images, labels, _ in data_loader:
        outputs = model(images)
        #print("Predictions:", outputs)
        #print("Ground Truth:", labels)

# Lists to store predictions and ground truth
all_predictions = []
all_ground_truth = []

# Collect predictions and ground truth
model.eval()
with torch.no_grad():
    for images, labels, _ in data_loader:
        outputs = model(images)
        all_predictions.append(outputs.numpy())
        all_ground_truth.append(labels.numpy())

# Convert lists to NumPy arrays
all_predictions = np.concatenate(all_predictions, axis=0)
all_ground_truth = np.concatenate(all_ground_truth, axis=0)

# Calculate metrics
mse = mean_squared_error(all_ground_truth, all_predictions)
mae = mean_absolute_error(all_ground_truth, all_predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Save predictions and ground truth to a CSV file
results_df = pd.DataFrame(
    np.hstack([all_ground_truth, all_predictions]),
    columns=["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3",
             "Pred_Class2.1", "Pred_Class2.2", "Pred_Class7.1", "Pred_Class7.2", "Pred_Class7.3"]
)
results_df.to_csv("data/Exercise_2/results.csv", index=False)
print("Results saved to data/Exercise_2/results.csv")

# Define the device (CPU or GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the device
#model.to(device)
# Evaluate the model
#evaluate_model(model, data_loader, device)

# Create a subfolder for this run
run_folder = os.path.join(folder_evaluation_2, f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# Plot predictions vs. ground truth for Q2 (Class2.1 and Class2.2)
plt.figure(figsize=(10, 5))
plt.scatter(all_ground_truth[:, 0], all_predictions[:, 0], label="Class2.1", alpha=0.7)
plt.scatter(all_ground_truth[:, 1], all_predictions[:, 1], label="Class2.2", alpha=0.7)
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.title("Predictions vs. Ground Truth for Q2")
#plt.savefig(folder_evaluation_2 + f"predictions_vs_ground_truth_q2_{timestamp}.png")  # Add timestamp
plt.savefig(os.path.join(run_folder, "predictions_vs_ground_truth_q2.png"))
# plt.show()

# Plot predictions vs. ground truth for Q7 (Class7.1, Class7.2, Class7.3)
plt.figure(figsize=(10, 5))
plt.scatter(all_ground_truth[:, 2], all_predictions[:, 2], label="Class7.1", alpha=0.7)
plt.scatter(all_ground_truth[:, 3], all_predictions[:, 3], label="Class7.2", alpha=0.7)
plt.scatter(all_ground_truth[:, 4], all_predictions[:, 4], label="Class7.3", alpha=0.7)
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Ground Truth")
plt.ylabel("Predictions")
plt.title("Predictions vs. Ground Truth for Q7")
plt.legend()
#plt.savefig(folder_evaluation_2 + f"predictions_vs_ground_truth_q7_{timestamp}.png")  # Add timestamp
plt.savefig(os.path.join(run_folder, "predictions_vs_ground_truth_q7.png"))
# plt.show()

# Save predictions and ground truth to a CSV file
results_df = pd.DataFrame(
    np.hstack([all_ground_truth, all_predictions]),
    columns=["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3",
             "Pred_Class2.1", "Pred_Class2.2", "Pred_Class7.1", "Pred_Class7.2", "Pred_Class7.3"]
)
results_df.to_csv("data/exercise_2/results.csv", index=False)
print("Results saved to data/exercise_2/results.csv")

# Calculate evaluation metrics
mse = mean_squared_error(all_ground_truth, all_predictions)
mae = mean_absolute_error(all_ground_truth, all_predictions)
r2 = r2_score(all_ground_truth, all_predictions)

# Print metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save metrics to a file
metrics_file = os.path.join(run_folder, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
print(f"Metrics saved to {metrics_file}")

# Plot Predictions vs. Ground Truth for each target
for i, target_name in enumerate(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]):
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

# Plot Residuals for each target
for i, target_name in enumerate(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]):
    residuals = all_predictions[:, i] - all_ground_truth[:, i]
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, label=target_name)
    plt.axvline(0, color="red", linestyle="--", label="Zero Residual")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residuals for {target_name}")
    plt.legend()
    residuals_file = os.path.join(run_folder, f"residuals_{target_name}.png")
    plt.savefig(residuals_file)
    print(f"Residuals plot saved to {residuals_file}")
    plt.close()

# Check which model performs worst
# Calculate per-class errors
per_class_mse = []
per_class_mae = []

for i, target_name in enumerate(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]):
    mse = mean_squared_error(all_ground_truth[:, i], all_predictions[:, i])
    mae = mean_absolute_error(all_ground_truth[:, i], all_predictions[:, i])
    per_class_mse.append(mse)
    per_class_mae.append(mae)
    print(f"{target_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")

# Identify the worst-performing class
worst_mse_class = np.argmax(per_class_mse)
worst_mae_class = np.argmax(per_class_mae)

print(f"\nWorst-performing class by MSE: Class {['Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3'][worst_mse_class]} with MSE: {per_class_mse[worst_mse_class]:.4f}")
print(f"Worst-performing class by MAE: Class {['Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3'][worst_mae_class]} with MAE: {per_class_mae[worst_mae_class]:.4f}")

# Save per-class errors to a file
error_file = os.path.join(run_folder, "per_class_errors.txt")
with open(error_file, "w") as f:
    for i, target_name in enumerate(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]):
        f.write(f"{target_name} - MSE: {per_class_mse[i]:.4f}, MAE: {per_class_mae[i]:.4f}\n")
    f.write(f"\nWorst-performing class by MSE: Class {['Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3'][worst_mse_class]} with MSE: {per_class_mse[worst_mse_class]:.4f}\n")
    f.write(f"Worst-performing class by MAE: Class {['Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3'][worst_mae_class]} with MAE: {per_class_mae[worst_mae_class]:.4f}\n")
print(f"Per-class errors saved to {error_file}")

# Bar plot for per-class MSE
plt.figure(figsize=(10, 5))
plt.bar(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"], per_class_mse, color="skyblue")
plt.xlabel("Class")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Per-Class MSE")
plt.savefig(os.path.join(run_folder, "per_class_mse.png"))
plt.close()

# Bar plot for per-class MAE
plt.figure(figsize=(10, 5))
plt.bar(["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"], per_class_mae, color="lightgreen")
plt.xlabel("Class")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Per-Class MAE")
plt.savefig(os.path.join(run_folder, "per_class_mae.png"))
plt.close()