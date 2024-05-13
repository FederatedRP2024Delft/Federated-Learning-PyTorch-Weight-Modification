import os
import sys

from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
#https://avandekleut.github.io/vae/
# https://kvfrans.com/deriving-the-kl/
# https://2020machinelearning.medium.com/exploring-different-methods-for-calculating-kullback-leibler-divergence-kl-in-variational-12197138831f
# https://stats.stackexchange.com/questions/562374/implementing-a-vae-in-pytorch-extremely-negative-training-loss
# https://stackoverflow.com/questions/74865368/kl-divergence-loss-equation
MNIST_INPUT_SIZE = 784
HIDDEN_LAYER_SIZE_1 = 512
HIDDEN_LAYER_SIZE_2 = 256
DEFAULT_LATENT_DIM = 10


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=DEFAULT_LATENT_DIM):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(MNIST_INPUT_SIZE, HIDDEN_LAYER_SIZE_1)  # 784 -> 512
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE_1, HIDDEN_LAYER_SIZE_2)  # 512 -> 256
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE_2, latent_dims)  # 256 -> 2
        self.linear4 = nn.Linear(HIDDEN_LAYER_SIZE_2, latent_dims)  # 256 -> 2

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # sample on GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        # (batch size, 2)
        log_var = torch.log(sigma ** 2)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return z




class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims=DEFAULT_LATENT_DIM):
        super(VariationalDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, HIDDEN_LAYER_SIZE_2)
        self.linear2 = nn.Linear(HIDDEN_LAYER_SIZE_2, HIDDEN_LAYER_SIZE_1)
        self.linear3 = nn.Linear(HIDDEN_LAYER_SIZE_1, MNIST_INPUT_SIZE)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=DEFAULT_LATENT_DIM):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def train_model(
            self,
            training_data,
            batch_size,
            epochs,
            beta=1.0
    ) -> tuple[list, list, list]:
        vae = self.to('cuda')
        vae.train()
        opt = torch.optim.Adam(params=vae.parameters())

        training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        total_across_epochs = []
        mse_across_epochs = []
        kl_across_epochs = []

        for epoch_idx in range(1, epochs + 1):
            batch_total_loss = 0
            batch_mse_loss = 0
            batch_kl_loss = 0
            num_batches = 0
            for batch_idx, (x, _) in enumerate(training_dataloader):
                num_batches += 1
                x = x.to('cuda')
                opt.zero_grad()

                x = torch.flatten(x, start_dim=1)
                # Forward pass
                x_hat = vae(x)


                # Calculate losses
                loss_fn = nn.MSELoss(reduction="none")
                mse_loss = torch.sum(loss_fn(x_hat,x),dim=1).mean()
                kl_loss = beta * vae.encoder.kl
                loss = mse_loss + kl_loss

                # Add losses to lists
                batch_total_loss += loss.item()
                batch_mse_loss += mse_loss.item()
                batch_kl_loss += kl_loss.item()

                # Backprop
                loss.backward()
                opt.step()

            print(
                f"Finished local epoch {epoch_idx} out of {epochs}, average loss across batches: {batch_total_loss / num_batches}")
            total_across_epochs.append(batch_total_loss / num_batches)
            mse_across_epochs.append(batch_mse_loss / num_batches)
            kl_across_epochs.append(batch_kl_loss / num_batches)

        return total_across_epochs, mse_across_epochs, kl_across_epochs

    def evaluate_model(
            self,
            validation_data,
            batch_size,
            beta=1.0
    ) -> tuple[float, float, float]:
        vae = self.to('cuda')
        vae.eval()

        validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

        total_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(validation_dataloader):
                num_batches += 1
                x = x.to('cuda')

                # Forward pass
                x = torch.flatten(x, start_dim=1)
                x_hat = vae(x)

                # Calculate losses
                loss_fn = nn.MSELoss(reduction="none")
                mse_loss = torch.sum(loss_fn(x_hat,x),dim=1).mean()
                kl_loss = beta * vae.encoder.kl
                loss = mse_loss + kl_loss

                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_kl_loss += kl_loss.item()

        return total_loss / num_batches, total_mse_loss / num_batches, total_kl_loss / num_batches



