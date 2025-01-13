import torch
from torch.utils.data import DataLoader
import os

from opts import args
from model import get_model
from dataset import get_dataset
import losses


def train():
    model.train()
    best_val_loss = float('inf')
    for epoch in range(args.n_epochs):
        total_loss = 0
        for itr, data in enumerate(train_dataloader):
            # Get data from dataloader
            y = data
            y = y.to(device)
            y.requires_grad = True

            # Forward pass
            y_pred, mean, log_var = model(y)

            # Calculate loss
            loss = losses.VAE_loss(y_pred, y, mean, log_var, args.latent_dim)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print training stats
            display_step = 500
            if itr and itr % display_step == 0:
                print(f"Epoch: {epoch} [{itr}/{len(train_dataloader)}] \t Loss: {total_loss / display_step}")
                total_loss = 0

        # Validate
        val_loss = val()

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = {
                'model': model.state_dict(),
                'opt': optimizer.state_dict()
            }
            torch.save(state_dict, os.path.join(args.checkpoint_dir, args.exp_name, f"epoch_{epoch}_loss{val_loss}.pt"))


def val():
    total_loss = 0
    for itr, data in enumerate(val_dataloader):
        # Get data from dataloader
        y = data
        y = y.to(device)

        # Forward pass
        with torch.no_grad():
            y_pred, mean, log_var = model(y)
        
        # Calculate loss
        loss = losses.VAE_loss(y_pred, y, mean, log_var, args.latent_dim)
        total_loss += loss.item()

    print(f"Val Loss: {total_loss / len(val_dataloader)}")
    return total_loss / len(val_dataloader)

def test():
    model.eval()
    for itr, data in enumerate(test_dataloader):
        # Get data from dataloader
        y = data
        y = y.to(device)

        # Forward pass
        with torch.no_grad():
            y_pred, mean, log_var = model(y)
        
        # Calculate loss
        loss = losses.VAE_loss(y_pred, y, mean, log_var, args.latent_dim)
        total_loss += loss.item()

        # Calculate metrics
        ...


if __name__ == '__main__':
    os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.best_checkpoint:
        state_dict = torch.load(os.path.join(args.checkpoint_dir, args.exp_name, args.best_checkpoint))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['opt'])

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
