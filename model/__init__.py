def get_model(args):
    if args.model == 'VAE':
        from .VariationalAutoencoder import VAE as Model
    elif args.model == 'LSTM':
        from .lstm import VAE as Model
    
    model = Model(args)
    return model
