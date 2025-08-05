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
