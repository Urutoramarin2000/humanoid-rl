import torch.nn as nn
import torch
import numpy as np
from .utils import get_activation, check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional
from torchsummary import summary

class VAE(nn.Module):
    def __init__(
        self,
        num_obs,
        num_history,
        num_latent,
        activation="elu",
        encoder_hidden_dims=[512, 256,256],
        decoder_hidden_dims=[512, 256, 128],
        device = "cuda"
    ):
        super().__init__()
        self.device = device
        self.num_obs = num_obs
        self.num_his = num_history
        self.num_latent = num_latent

        self.encoder = encoder(
            num_history * num_obs,
            self.num_latent * 2 + 6,
            activation,
            encoder_hidden_dims,
        )

        num_vel = 3
        self.decoder = decoder(
            self.num_latent + num_vel,
            num_obs,# num_obs
            activation,
            decoder_hidden_dims,
        )

        # self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        # self.latent_var = nn.Linear(num_latent * 4, num_latent)
        # self.vel_mu = nn.Linear(num_latent * 4, 3)
        # self.vel_var = nn.Linear(num_latent * 4, 3)

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        # latent_mu = self.latent_mu(encoded)
        # latent_var = self.latent_var(encoded)
        # vel_mu = self.vel_mu(encoded)
        # vel_var = self.vel_var(encoded)
        latent_mu = encoded[:,:self.num_latent]
        latent_var = encoded[:,self.num_latent:self.num_latent * 2]
        vel_mu = encoded[:,self.num_latent * 2:self.num_latent * 2 + 3]
        vel_var = encoded[:,self.num_latent * 2 + 3:self.num_latent * 2 + 6]
        return [latent_mu, latent_var, vel_mu, vel_var]

    def decode(self, z, v):
        input = torch.cat([z, v], dim=1)
        output = self.decoder(input)
        return output

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, cv) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return: eps * std + mu
        """
        std = torch.exp(0.5 * logvar) * (1 - np.tanh(cv))
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs_history, cv):
        latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
        latent_var = torch.clip(latent_var, -32767.0 , 5.0)
        z = self.reparameterize(latent_mu, latent_var, cv)
        # vel做一个mseloss与decoder latent作的loss相加
        # vel = self.reparameterize(vel_mu, vel_var, cv)
        vel = vel_mu.clone()
        return [z, vel], [latent_mu, latent_var, vel_mu, vel_var]

    def loss_fn(self, obs_history, obs_next, vel, cv, kld_weight=1.0):
        [z, v], [latent_mu, latent_var, vel_mu, vel_var] = self.forward(obs_history, cv)

        # Body velocity estimation loss
        vel_loss = functional.mse_loss(v, vel, reduction="none").mean(-1)
        v_v = v
        v_vel = vel
        v_Difference = v_v.cpu().detach().numpy() - v_vel.cpu().detach().numpy()
        v_avg_diff_x ,v_avg_diff_y,v_avg_diff_z= np.mean(np.abs(v_Difference),axis=0)


        # MSE of obs in VAE loss
        recons_obs = self.decode(z, v)
        recons_loss = functional.mse_loss(recons_obs, obs_next, reduction="none").mean( #obs_next
            -1
        )

        # KL divergence as latent loss
        # KL in VAE = ( exp(var) - (1+var) + mu^2 )
        kld_loss = -0.5 * torch.sum(
            1 + latent_var - latent_mu**2 - latent_var.exp(), dim=1
        )

        loss = recons_loss + vel_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "recons_loss": recons_loss,
            "vel_loss": vel_loss,
            "kld_loss": kld_loss,
            "v_avg_diff_x": v_avg_diff_x,
            "v_avg_diff_y": v_avg_diff_y,
            "v_avg_diff_z": v_avg_diff_z,
        }

    def sample(self, obs_history, cv):
        """
        :return estimation = [z, vel]
        :dim(z) = num_latent
        :dim(vel) = 3
        """
        estimation, output = self.forward(obs_history, cv)
        return estimation,output

    def inference(self, obs_history):
        """
        return [latent_mu, vel_mu]
        """
        [z, v], [latent_mu, latent_var, vel_mu, vel_var] = self.forward(obs_history, cv=0)
        # _, latent_params = self.forward(obs_history, cv = 0)
        # latent_mu, latent_var, vel_mu, vel_var = latent_params
        return [z ,v ,latent_mu, vel_mu]


class encoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(RS_obs_history)


class decoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.decoder = nn.Sequential(*module)

    def forward(self, input):
        return self.decoder(input)
