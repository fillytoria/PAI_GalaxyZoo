import os
import torch
import torch.nn as nn 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.optim import Adam
from dataclasses import dataclass

import datetime #save plots with timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

# Define regression model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Change input channels to 1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 5)  # Output size matches the number of regression targets

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x    


def main(dataset, folder_evaluation):
    cli = simple_parsing.parse(TrainingCli)
    config = prepare_config(
        Path(f"outputs/{cli.run_name}/config.yaml"),
        Path("config_default.yaml"),
        cli.run_name,
        not cli.no_config_edit,
    )

    # Initialize model, loss function, and optimizer
    model = RegressionModel()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    num_epochs = 10
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
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for images, labels, _ in data_loader:
            outputs = model(images)
            print("Predictions:", outputs)
            print("Ground Truth:", labels)

    # Initialize lists to store predictions and ground truth
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
        columns=["GT_Class2.1", "GT_Class2.2", "GT_Class7.1", "GT_Class7.2", "GT_Class7.3",
                 "Pred_Class2.1", "Pred_Class2.2", "Pred_Class7.1", "Pred_Class7.2", "Pred_Class7.3"]
    )
    results_df.to_csv("data/exercise_2/results.csv", index=False)
    print("Results saved to data/exercise_2/results.csv")

    return regression_results, all_ground_truth, all_predictions, run_folder

if __name__ == "__main__":
    logging.basicConfig()
    main()    
