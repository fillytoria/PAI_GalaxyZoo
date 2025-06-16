import os
import shutil
import torch
from glob import glob

from galaxy_classification.data import get_labels_train

def setup_folders(folder_name: str):
    '''Set up extra folders for the current script. 
    Load images and labels from data folder.'''
    folder = os.path.join("data", folder_name)
    folder_images = os.path.join(folder, "images")
    file_labels = os.path.join(folder, "extracted_labels.csv")

    os.makedirs(folder_images, exist_ok=True)
    
    labels_file = "data/training_solutions_rev1.csv"
    labels = get_labels_train(labels_file)  # Load labels from CSV file

    image_folder = "data/images/"
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))

    folder_evaluation = os.path.join(folder, "evaluation/")
    
    return folder, folder_images, file_labels, labels, image_paths, folder_evaluation

def condition(labels, column_range):
    '''Check if any of the first three label values are greater than or equal to 0.8 (default threshold). 
    If yes, return True.'''
    condition = labels[:, column_range[0]:column_range[1]] >= 0.8
    return torch.sum(condition, dim=1).bool()

def one_hot_encode(labels, threshold=0.8):
    '''Convert labels into one-hot encoded vectors based on a threshold.'''
    one_hot = torch.zeros_like(labels)
    one_hot[labels >= threshold] = 1   # Set to 1 where labels meet the threshold
    return one_hot