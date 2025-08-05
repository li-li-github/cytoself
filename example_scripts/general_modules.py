"""
###################
General Data Loader

The CytoselfFullTrainer module works with standard torch DataLoader objects, provided that they contain a few
core components.

Requirements:
1. Batches must return a dictionary with keys 'image' and 'label'.
  - 'image' should be a tensor of shape (batch_size, channels, height, width).
    - for the cytoself paper, "image" has 3 channels (protein, nucleus, nuclear distance transform)
  - 'label' should be an integer or a one-hot encoded vector of shape (batch_size, num_classes).
2. dataset must have a `unique_labels` attribute that is a list of unique labels in the dataset.

###################
"""

from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset


class GeneralDataset(Dataset):

    def __init__(
        self,
        unique_labels: Sequence[str],
    ):

        self.transform: Optional[Sequence] = None
        self.unique_labels = unique_labels

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        array = None  # Load data however you like

        label_int = None  # load the labels however you like

        if self.transform is not None:
            data = self.transform(array)
        return {"image": data, "label": label_int}

# Define a Dataloader based on the above dataset
loader = torch.utils.data.DataLoader(
    dataset=None, # instance of GeneralDataset
    batch_size=None,
)

"""
####################
General Data Manager
####################

This module defines the minimum requirements for a data manger to be used with the `CytoselfFullTrainer` module.

Requirements:
1. A class that inherits from `DataManagerBase`.
2. The object must have attributes `train_loader`, `val_loader`, and `test_loader` which are instances of `torch.utils.data.DataLoader`.
3. The object must have attributes `train_variance`, `val_variance`, and `test_variance` which are floats representing the variance of 
the training, validation, and test datasets respectively. This is essential for the training of a VQVAE model.

"""

import numpy as np

from cytoself.datamanager.base import DataManagerBase


class CustomDataManager(DataManagerBase):
    def __init__(
        self,
        train_loader=None,
        val_loader=None,
        test_loader=None,
    ):
        """
        Variance of the dataset is essential to training a VQVAE
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.unique_labels = None

        self.train_variance = None
        self.test_variance = None
        self.val_variance = None
        if train_loader is not None and self.train_variance is None:
            self.train_variance = self._estimate_var_from_loader(train_loader)
        if val_loader is not None and self.val_variance is None:
            self.val_variance = self._estimate_var_from_loader(val_loader)
        if test_loader is not None and self.test_variance is None:
            self.test_variance = self._estimate_var_from_loader(test_loader)

    def _estimate_var_from_loader(self, loader, num_batches: int = 10):
        count = 0
        mean = 0.0
        M2 = 0.0

        for i, batch in enumerate(loader):
            data = batch["image"].detach().numpy().astype(np.float32).ravel()

            batch_n = data.size
            batch_mean = data.mean()
            batch_M2 = ((data - batch_mean) ** 2).sum()

            # Welford update
            if count == 0:
                mean = batch_mean
                M2 = batch_M2
                count = batch_n
            else:
                delta = batch_mean - mean
                total_count = count + batch_n
                M2 = M2 + batch_M2 + delta**2 * count * batch_n / total_count
                count = total_count

            if i >= num_batches:
                break

        variance = M2 / count if count > 0 else float("nan")
        return variance
