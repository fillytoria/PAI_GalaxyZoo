from dataclasses import dataclass
from glob import glob
import os
import torch
import pandas as pd

from numpy.typing import NDArray
from pathlib import Path
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from typing import Self, cast

from PIL import Image

class GalaxyZooDataset(Dataset):
    def __init__(self, image_paths, labels, max_samples=None, preprocessor=None, transform = None):
        self.image_paths = image_paths[:max_samples] if max_samples else image_paths
        self.labels = labels[:max_samples] if max_samples else labels
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.transform = transform #des eig nur für augmentation nach exercise 3, wenn i des doch net mach wegtun

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        label = self.labels[index]

        # Load and preprocess the image
        image = Image.open(file_path).convert('RGB')
        image = self.preprocessor.preprocess(image)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:  #des ebenfalls nur für des augmentation in exercise_3
            image = self.transform(image)

        return image, label_tensor, file_path        


@dataclass
class ImagePreprocessor:
    def __init__(self):
        # Define image transform: crop central 207x207 and resize to 64x64
        self.transform = transforms.Compose([
            transforms.CenterCrop(207),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.to_pil = ToPILImage()
    def preprocess(self, image):
        """
        Apply the defined transformations to the input image.
        :param image: Input image (PIL Image or Tensor)
        :return: Transformed image (Tensor)
        """
        return self.transform(image)


@dataclass
class SplitDataLoader:
    training_dataloader: DataLoader
    validation_dataloader: DataLoader

    def __init__(
        self,
        dataset_prepared: Dataset,
        validation_fraction: float,   # Fraction of data for validation, 20% by default
        batch_size: int,            # Batch size for training and validation
    ):
        # Calculate sizes for training and validation splits
        validation_size = int(validation_fraction * len(dataset_prepared))
        train_size = len(dataset_prepared) - validation_size

        # Split the dataset into training and validation sets
        training_dataset, validation_dataset = (
            torch.utils.data.random_split(
                dataset_prepared,
                lengths=[train_size, validation_size],
                generator=Generator().manual_seed(42),
            )
        )

        # Create DataLoaders for training and validation
        self.training_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True
        )

        # Print the number of samples in each split
        print(f"Training samples: {train_size}, Validation samples: {validation_size}")


# Load label file
def get_labels_train(file_galaxy_labels) -> torch.Tensor:
    df_galaxy_labels = pd.read_csv(file_galaxy_labels)
    #return df_galaxy_labels
    labels_array = df_galaxy_labels.values  # Convert DataFrame to NumPy array
    return torch.tensor(labels_array, dtype=torch.float32)  # Convert to PyTorch tensor
