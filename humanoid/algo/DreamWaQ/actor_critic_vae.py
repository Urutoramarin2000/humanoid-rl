import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import (
    get_activation,
    MultivariateGaussianDiagonalCovariance,
    init_orhtogonal,
)
from .state_estimator import VAE

#from graphviz import Digraph
from torch.autograd import Variable, Function

class ActorCritic_VAE(nn.Module):
    def __init__(
        self,
        num_obs,
        num_privileged_obs,
        num_actions=12,
        num_history=14,
        num_latent=16,
        num_est_prob=3,
        activation="elu",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[256, 128, 256, 64],
        decoder_hidden_dims=[64, 128, 256],
        init_noise_std=1.0,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.vae = VAE(
            num_obs=num_obs,
            num_history=num_history,
            num_latent=num_latent,
            activation=self.activation,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        print(self.vae)
        self.actor = Actor(
            input_size=num_obs + num_latent + num_est_prob,    #disable VAE
            # input_size = num_obs,
            output_size=num_actions,
            activation=self.activation,
            hidden_dims=actor_hidden_dims,
        )
        """
        :input = obs + latent + estimated_vel
        :output = actions
        """
        print(self.actor)
        real_vel = 3
        self.critic = Critic(
            # input_size=num_obs + real_vel + num_privileged_obs,
            input_size = num_privileged_obs, #+ real_vel,
            output_size=1,
            activation=self.activation,
            hidden_dims=critic_hidden_dims,
        )
        """
        :input = obs + real_vel + privileged_obs
        :output = reward
        """
        print(self.critic)
        self.distribution = None
        
        # self.distribution = MultivariateGaussianDiagonalCovariance(
        #     dim=num_actions,
        #     init_std=init_noise_std,
        # )
        # """distribution of noise for action"""
        # 在确定代码无误后，禁用参数验证可以略微提高性能
        Normal.set_default_validate_args = False

        self.vae.apply(init_orhtogonal)
        # self.actor.apply(init_orhtogonal)
        # self.critic.apply(init_orhtogonal)

    #! rollout 的时候需要随机性, 这里是 bootstrap
    def act_student(self, obs, obs_history, cv):
        """
        :obs_dict: obs, obs_history
        :return distribution.sample()
        :其中std会在模型的训练过程中自动调整
        """
        # disable vae
        [z, vel],[latent_mu, latent_var, vel_mu, vel_var] = self.vae.sample(obs_history, cv)
        latent_and_estimatedVEL = torch.cat([z, vel], dim=1)
        input = torch.cat([obs,latent_and_estimatedVEL],dim=1)
        if torch.isnan(latent_and_estimatedVEL).any():
            raise ValueError("CENet ocurrs nan")
        self.pre_out = latent_and_estimatedVEL.detach()
        self.pre_latent_mu = latent_mu.detach()
        self.pre_latent_var = latent_var.detach()
        self.pre_vel_mu = vel_mu.detach()
        self.pre_vel_var = vel_var.detach()
        # input = obs
        self.pre_std = self.std.detach()
        action_mean = self.actor.forward(input)
        if torch.isnan(action_mean).any():
            raise ValueError("action_mean ocurrs nan")
        self.update_distribution(action_mean)
        return self.distribution.sample()


    def reset(self, dones=None):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # eval和play模式用的，推理不用
    def act_inference(self, obs_input, obs_history):
        z ,vel,latent_mu, vel_mu = self.vae.inference(obs_history)
        latent = torch.cat([latent_mu, vel_mu], dim=1)
        print('v_est',vel.cpu().detach().numpy())
        
        input = torch.cat([obs_input, latent],dim=1)
        return self.actor.forward(input)

    def evaluate(self, obs, privileged_observations):
        input = privileged_observations
        value = self.critic.forward(input)
        return value
    
    def update_distribution(self, mean):
        self.distribution = Normal(mean, mean*0. + self.std)

class Actor(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
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
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


class Critic(nn.Module):
    def __init__(
        self, input_size, output_size, activation, hidden_dims=[512, 256, 128]
    ):
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
        self.net = nn.Sequential(*module)

    def forward(self, input):
        return self.net(input)


