import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from galaxy_classification.data import GalaxyZooDataset
from galaxy_classification.utils import setup_folders

from regression import RegressionModel, regression_main

folder_name = "Exercise3_RegressionQ6Q8"
folder_3, folder_images_3, file_labels_3, labels, image_paths, folder_evaluation_3 = setup_folders(folder_name)

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

# Ensure consistent slicing
max_samples = 500
image_paths = image_paths[:max_samples]
labels = labels[:max_samples]

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=max_samples)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train and evaluate the Q2Q7 model
print("\nTraining Q2Q7 Model...")
q2q7_model = RegressionModel(input_size=64 * 64, output_size=5)  # Adjust input/output sizes as needed
q2q7_results, q2q7_ground_truth, q2q7_predictions, q2q7_run_folder = regression_main(
    model=q2q7_model,
    dataset=dataset,
    folder_evaluation=folder_evaluation_3,
    target_columns=list(range(5))  # Q2 and Q7 columns
)

# Train and evaluate the Q6Q8 model using Q2Q7 predictions as input
print("\nTraining Q6Q8 Model...")
q6q8_model = RegressionModel(input_size=64 * 64 + 5, output_size=9)  # Include Q2Q7 predictions as input
q6q8_results, q6q8_ground_truth, q6q8_predictions, q6q8_run_folder = regression_main(
    model=q6q8_model,
    dataset=dataset,
    folder_evaluation=folder_evaluation_3,
    target_columns=list(range(5, 14)),  # Q6 and Q8 columns
    additional_features=q2q7_predictions  # Use Q2Q7 predictions as additional input
)

# Save metrics for Q6Q8 model
q6q8_metrics_file = os.path.join(q6q8_run_folder, "q6q8_metrics.txt")
with open(q6q8_metrics_file, "w") as f:
    f.write(f"Q6Q8 Model - Mean Squared Error (MSE): {q6q8_results['mse']:.4f}\n")
    f.write(f"Q6Q8 Model - Mean Absolute Error (MAE): {q6q8_results['mae']:.4f}\n")
    f.write(f"Q6Q8 Model - R² Score: {q6q8_results['r2']:.4f}\n")
print(f"Q6Q8 metrics saved to {q6q8_metrics_file}")

# Save plots for Q6Q8 model
target_names = [
    "Class6.1", "Class6.2",
    "Class8.1", "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6", "Class8.7"
]
for i, target_name in enumerate(target_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(q6q8_ground_truth[:, i], q6q8_predictions[:, i], alpha=0.7, label=target_name)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title(f"Q6Q8 Predictions vs. Ground Truth for {target_name}")
    plt.legend()
    q6q8_plot_file = os.path.join(q6q8_run_folder, f"q6q8_predictions_vs_ground_truth_{target_name}.png")
    plt.savefig(q6q8_plot_file)
    print(f"Q6Q8 plot saved to {q6q8_plot_file}")
    plt.close()