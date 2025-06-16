import os
import torch 
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import DataLoader

from digit_classification.data import GalaxyZooDataset, get_labels_train

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime # Save plots with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

from galaxy_classification.utils import setup_folders

from regression import main as regression_main

# Paths
folder_name = "Exercise2_RegressionQ2Q7"
folder_2, folder_images_2, file_labels_2, labels, image_paths, folder_evaluation_2 = setup_folders(folder_name)

# Extract relevant columns for regression, [Class2.1, Class2.2] for Q2, [Class7.1, Class7.2, Class7.3] for Q7
q2_columns = [5, 6]  
q7_columns = [17, 18, 19]  
relevant_columns = q2_columns + q7_columns
labels = labels[:, relevant_columns]  # Extract only the relevant columns

# Save relevant labels to a new CSV file
relevant_labels_df = pd.DataFrame(
    labels.numpy(),  # Convert tensor to NumPy array
    columns=["Class2.1", "Class2.2", "Class7.1", "Class7.2", "Class7.3"]  # Column names
)
relevant_labels_df.to_csv(file_labels_2, index=False)
print(f"Relevant labels saved to {file_labels_2}")

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=1000)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Perform regression task
for images, labels, paths in data_loader:
    q2_labels = labels[:, :2]  # First two columns are Q2 (Class2.1, Class2.2)
    q7_labels = labels[:, 2:]  # Last three columns are Q7 (Class7.1, Class7.2, Class7.3)

    # Normalize Q2 and Q7 labels to ensure sums are 1
    #q2_labels = q2_labels / torch.sum(q2_labels, dim=1)
    #q7_labels = q7_labels / torch.sum(q7_labels, dim=1)

    # check sums
    q2_sums = torch.sum(q2_labels, dim=1)
    q7_sums = torch.sum(q7_labels, dim=1)

    # Check hierarchical structure
    # Load the original Q1.1 and Q1.2 values from the original labels
    #original_labels = get_labels_train(labels_file)  # Load full labels
    #q1_1_values = original_labels[:, 2]  # Third column is Q1.1
    #q1_2_values = original_labels[:, 1]  # Second column is Q1.2
    #galaxy_id = original_labels[:, 0]  # First column is Galaxy ID

regression_results, all_ground_truth, all_predictions, run_folder = regression_main(dataset=dataset, folder_evaluation=folder_evaluation_2)

# Check for which question the model does worst
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
