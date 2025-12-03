import torch
from torch.utils.data import DataLoader
import os

from opts import args
from model import get_model
from model.discriminator import Discriminator
from dataset import get_dataset
from losses import VAE_loss, shuffle, adversarial_loss, classification_loss


def train():
    model.train()
    best_val_loss = float('inf')
    for epoch in range(args.n_epochs):
        total_vae_loss = 0
        total_genre_loss = 0
        total_gen_loss = 0
        total_disc_loss = 0
        for itr, data in enumerate(train_dataloader):
            # Get data from dataloader
            y, genre = data
            y = y.to(device)
            genre = genre.reshape(-1).to(device)


            ### Train discriminator
            disc_optimizer.zero_grad()

            # Get VAE outputs
            y_pred, mean_genre, log_var_genre, mean_instance, log_var_instance = model(y)

            # Shuffle pairs
            n = mean_genre.shape[0]
            shuffled_genre = shuffle(mean_genre)
            # Feed into discriminator network
            shuffle_pred = discriminator(torch.cat((shuffled_genre, mean_instance), dim=-1).detach())
            unshuffle_pred = discriminator(torch.cat((mean_genre, mean_instance), dim=-1).detach()) 
            # Get adversarial loss
            shuffle_loss = adversarial_loss(shuffle_pred, torch.zeros_like(shuffle_pred))
            unshuffle_loss = adversarial_loss(unshuffle_pred, torch.ones_like(unshuffle_pred))
            adv_loss = (shuffle_loss + unshuffle_loss) / 2

            # Update discriminator
            adv_loss.backward()
            disc_optimizer.step()


            ### Train VAE
            vae_optimizer.zero_grad()

            # Get VAE outputs
            y_pred, mean_genre, log_var_genre, mean_instance, log_var_instance = model(y)

            # Feed into discriminator network
            unshuffle_pred = discriminator(torch.cat((mean_genre, mean_instance), dim=-1))  # Note that we don't detach here 
            # Get adversarial loss
            unshuffle_loss = adversarial_loss(unshuffle_pred, torch.zeros_like(unshuffle_pred))

            # Get genre prediction
            genre_pred = model.genre_classifier(mean_genre)
            # Get genre loss
            genre_loss = classification_loss(genre_pred, genre) 

            # Get VAE loss for both genre and instance distributions
            vae_loss = VAE_loss(y_pred, y, mean_genre, log_var_genre, mean_instance, log_var_instance)

            # Update VAE
            loss = vae_loss + genre_loss + unshuffle_loss
            loss.backward()
            vae_optimizer.step()


            total_vae_loss += vae_loss.item()
            total_genre_loss += genre_loss.item()
            total_gen_loss += unshuffle_loss.item()
            total_disc_loss += adv_loss.item()

            # Print training stats
            display_step = 10
            if itr and itr % display_step == 0:
                print(f"\
                    Epoch: {epoch} [{itr}/{len(train_dataloader)}] \t \
                    VAE Loss: {total_vae_loss / display_step} \t \
                    Genre Loss: {total_genre_loss / display_step} \t \
                    Generator Loss: {total_gen_loss / display_step} \t \
                    Discriminator Loss: {total_disc_loss / display_step}\
                ")

                total_vae_loss = 0
                total_genre_loss = 0
                total_gen_loss = 0
                total_disc_loss = 0


        # Validate
        val_loss = val()

        # Save some samples
        torch.save(y, "samples/epoch_{epoch}_y.pt")
        torch.save(y_pred, "samples/epoch_{epoch}_y_pred.pt")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = {
                'model': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'vae_opt': vae_optimizer.state_dict(),
                'disc_opt': disc_optimizer.state_dict()
            }
            torch.save(state_dict, os.path.join(args.checkpoint_dir, args.exp_name, f"epoch_{epoch}_loss{val_loss}.pt"))


def val():
    total_loss = 0
    for itr, data in enumerate(val_dataloader):
        # Get data from dataloader
        y, genre = data
        y = y.to(device)
        genre = genre.to(device)

        # Forward pass
        with torch.no_grad():
            y_pred, mean_genre, log_var_genre, mean_instance, log_var_instance = model(y)
            genre_pred = model.genre_classifier(mean_genre)
        
        # Calculate loss
        vae_loss = VAE_loss(y_pred, y, mean_genre, log_var_genre, mean_instance, log_var_instance)
        genre_loss = classification_loss(genre_pred, genre) 
        loss = vae_loss + genre_loss  # Ignore adversarial losses for now
        total_loss += loss.item()

    print(f"Val Loss: {total_loss / len(val_dataloader)}")
    return total_loss / len(val_dataloader)


def test():
    model.eval()
    total_loss = 0
    for itr, data in enumerate(test_dataloader):
        # Get data from dataloader
        y, genre = data
        y = y.to(device)
        genre = genre.to(device)

        # Forward pass
        with torch.no_grad():
            y_pred, mean_genre, log_var_genre, mean_instance, log_var_instance = model(y)
            genre_pred = model.genre_classifier(mean_genre)
        
        # Calculate loss
        vae_loss = VAE_loss(y_pred, y, mean_genre, log_var_genre, mean_instance, log_var_instance)
        genre_loss = classification_loss(genre_pred, genre) 
        loss = vae_loss + genre_loss  # Ignore adversarial losses for now
        total_loss += loss.item()

    print("Total loss:", total_loss / len(test_dataloader))


if __name__ == '__main__':
    os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args).to(device)
    discriminator = Discriminator(latent_dim=args.latent_dim, hidden_dim=128).to(device)

    vae_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    if args.best_checkpoint:
        state_dict = torch.load(os.path.join(args.checkpoint_dir, args.exp_name, args.best_checkpoint))
        model.load_state_dict(state_dict['model'])
        discriminator.load_state_dict(state_dict['discriminator'])
        vae_optimizer.load_state_dict(state_dict['vae_opt'])
        disc_optimizer.load_state_dict(state_dict['disc_opt'])

    if args.mode == 'train':
        # Create train dataloader
        train_dataset = get_dataset(args, split='train')
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        # Create val dataloader
        val_dataset = get_dataset(args, split='train')
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=4
        )

        train()

    elif args.mode == 'test':
        # Create test dataloader
        test_dataset = get_dataset(args, split='test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size = args.batch_size,
            num_workers=4
        )

        test()
