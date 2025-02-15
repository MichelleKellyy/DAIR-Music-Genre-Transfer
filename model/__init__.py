def get_model(args):
    if args.model == 'VAE':
        from .lstm import VAE as Model
    
    model = Model(args)
    return model
