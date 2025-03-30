def get_dataset(args, split):
    if args.dataset == 'piano_roll':
        from .piano_roll_dataset import PianoRollDataset as Dataset
    elif args.dataset == 'REMI':
        from .remi_dataset import REMIDataset as Dataset

    dataset = Dataset(args, split)
    return dataset
