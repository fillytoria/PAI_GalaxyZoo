import os
import torch
import pandas as pd
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from galaxy_classification.data import GalaxyZooDataset, get_labels_train
from galaxy_classification.utils import setup_folders, condition, one_hot_encode

from train import main as train_main
from evaluate import main as evaluate_main

# Paths
folder_name = "Exercise1_ClassificationQ1"
folder_1, folder_images_1, file_labels_1, labels, image_paths, folder_evaluation_1 = setup_folders(folder_name)

# Create dataset and dataloader
dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels, max_samples=20000)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)

print("Dataset loaded with", len(dataset), "samples.")

# Loop through and save selected images and labels
files_1 = []
labels_1 = []

for images, labels, paths in data_loader:
    # Check condition for labels and get indices where condition is met
    cond = condition(labels, (1, 4))    
    indices = torch.where(cond)[0]      

    # Check each image and label
    for i in indices:
        selected_images = images[i]
        selected_labels = labels[i, 1:4]  # Select first three label values (corresponding to Class1.1, Class1.2, Class1.3)
        selected_labels = one_hot_encode(selected_labels)  # Convert to one-hot encoded labels
        
        # Save images for which condition is met
        filename = os.path.basename(paths[i])
        save_image(selected_images, os.path.join(folder_images_1, filename))

        # Save labels for which condition is met
        labels_1.append(selected_labels)  
        files_1.append(filename)

# Create and save the DataFrame for filtered labels
labels_df = pd.DataFrame(torch.stack(labels_1).numpy(), columns=["Type1.1", "Type1.2", "Type1.3"])
labels_df.insert(0, 'filename', files_1)  # Add filenames as first column
labels_df.to_csv(file_labels_1, index=False)

print(f"Saved {len(files_1)} images to {folder_images_1} and labels to {file_labels_1}.")

# Use the filtered files and labels
image_paths = [os.path.join(folder_images_1, filename) for filename in files_1]  # Use files_1 for image paths
labels = torch.stack(labels_1)

# Create dataset with only filtered images and labels
galaxy_dataset = GalaxyZooDataset(image_paths=image_paths, labels=labels)

evaluation_set, model = train_main(galaxy_dataset=galaxy_dataset, output_folder=folder_1)

evaluate_main(val_loader=evaluation_set, galaxy_dataset=galaxy_dataset, model = model, output_folder=folder_1)
