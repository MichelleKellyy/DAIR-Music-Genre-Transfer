import argparse

parser = argparse.ArgumentParser()


### Training arguments

parser.add_argument(
    '--exp_name',
    type=str,
    help="Experiment name"
)

parser.add_argument(
    '--mode',
    type=str,
    choices=['train', 'test'],
    help="Whether to train or test"
)

parser.add_argument(
    '--n_epochs',
    type=int,
    help="The number of training epochs"
)

parser.add_argument(
    '--lr',
    type=float,
    help="The learning rate for training"
)

parser.add_argument(
    '--batch_size',
    type=int,
    help="Training batch size"
)

parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default="./checkpoints",
    help="Directory for saving/loading checkpoints"
)

parser.add_argument(
    '--best_checkpoint',
    type=str,
    default="",
    help="File name of best checkpoint to load"
)

### Model arguments

parser.add_argument(
    '--model',
    type=str,
    choices=['VAE'],
    help="Model type"
)

parser.add_argument(
    '--latent_dim',
    type=int,
    help="Number of dimensions for VAE latent space"
)

### Dataset arguments

parser.add_argument(
    '--dataset',
    type=str,
    choices=['piano_roll'],
    help="Dataset to use"
)

parser.add_argument(
    '--data_dir',
    type=str,
    help="Path to data root directory"
)

args = parser.parse_args()
