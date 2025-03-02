def get_model(args):
    if args.model == 'VAE':
        from .VariationalAutoencoder import VAE as Model
    
    model = Model(args)
    return model
