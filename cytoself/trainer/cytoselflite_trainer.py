from typing import Optional, Union

from cytoself.trainer.autoencoder.cytoselflite import CytoselfLite
from cytoself.trainer.cytoselffull_trainer import CytoselfFullTrainer


class CytoselfLiteTrainer(CytoselfFullTrainer):
    """
    Trainer object for CytoselfLite model
    """

    def __init__(
        self,
        train_args: dict,
        homepath: str = './',
        device: Optional[Union[str, int]] = None,
        model: Optional = None,
        model_args: dict = None,
        use_mixed_precision: bool = True,
        use_multi_gpus: bool = False,
    ):
        """
        Initializes cytoself lite trainer

        Parameters
        ----------
        train_args : dict
            Arguments for training
        homepath : str
            Path where training results will be saved
        device : str
            Specify device; e.g. cpu, cuda, cuda:0 etc.
        model : Optional[torch.nn.Module]
            An autoencoder model class (uninitialized model)
        model_args : dict
            Arguments for model construction
        use_mixed_precision : bool
            Use mixed precision if True
        use_multi_gpus : bool
            Use multiple GPUs if True
        """
        if model is None:
            model = CytoselfLite
        super().__init__(train_args, homepath, device, model, model_args, use_mixed_precision, use_multi_gpus)
