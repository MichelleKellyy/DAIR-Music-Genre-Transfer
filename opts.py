import argparse

parser = argparse.ArgumentParser()


### Training arguments

parser.add_argument(
    '--exp_name',
    type=str,
    default='sample_experiment',
    help="Experiment name"
)

parser.add_argument(
    '--mode',
    type=str,
    #choices=['train', 'test'],
    default='train',
    help="Whether to train or test"
)

parser.add_argument(
    '--n_epochs',
    type=int,
    default=100,
    help="The number of training epochs"
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.000001,
    help="The learning rate for training"
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help="Training batch size"
)

parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default="C:/Users/Miche/Downloads/checkpoints",
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
    #choices=['VAE'],
    default='VAE',
    help="Model type"
)

parser.add_argument(
    '--latent_dim',
    type=int,
    default=128,
    help="Number of dimensions for VAE latent space"
)

### Dataset arguments

parser.add_argument(
    '--dataset',
    type=str,
    #choices=['piano_roll'],
    default='piano_roll',
    help="Dataset to use"
)

parser.add_argument(
    '--data_dir',
    type=str,
    default="C:/Users/Miche/Downloads/Piano_Rolls0/Piano_Rolls0",
    help="Path to data root directory"
)

args = parser.parse_args()
