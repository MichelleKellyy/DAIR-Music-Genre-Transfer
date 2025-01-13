def get_dataset(args, split):
    if args.dataset == 'piano_roll':
        from .piano_roll_dataset import PianoRollDataset as Dataset

    dataset = Dataset(args, split)
    return dataset
