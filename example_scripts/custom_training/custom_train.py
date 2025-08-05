from cytoself.datamanager.custom import CustomDataManager
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer


def train(
    output_path,
):
    # instances of torch.utils.data.DataLoader that meet the requirements outlined in example_scripts/general_modules.py
    train_loader = None
    val_loader = None
    test_loader = None

    manager = CustomDataManager(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )

    # Below parameters defined to match the Cytoself paper, but can be adjusted based on your dataset
    model_args = {
        "input_shape": (3, 100, 100),
        "emb_shapes": ((25, 25), (4, 4)),
        "output_shape": (3, 100, 100),
        "fc_output_idx": "all",
        "vq_args": {"num_embeddings": 2048, "embedding_dim": 64},
        "num_class": None,  # Set to the number of unique classes in your dataset
        "fc_input_type": "vqvec",
    }
    train_args = {
        "lr": 1e-4,
        "max_epoch": 500,
        "reducelr_patience": 6,
        "reducelr_increment": 0.1,
        "earlystop_patience": 30,
    }

    trainer = CytoselfFullTrainer(
        train_args, homepath=output_path, model_args=model_args
    )
    trainer.fit(manager, tensorboard_path="tb_logs")

    return


if __name__ == "__main__":
    output_path = "/path/to/results/"  # Change this
    train(output_path)
