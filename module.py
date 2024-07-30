# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


class Layer1(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class Layer2(nn.Module):
    
    def __init__(
        self,
        n_in: int = 128,
        n_out: int = 10,
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.var_eps = var_eps
        self.mean_encoder = nn.Linear(n_in, n_out)
        self.var_encoder = nn.Linear(n_in, n_out)
        self.var_activation = torch.exp if var_activation is None else var_activation
        
    def forward(self, x):
        q_m = self.mean_encoder(x)
        q_v = self.var_activation(self.var_encoder(x)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        
        return dist, latent


class Layer3(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
    
    def forward(self, x):
        return self.network(x)


class Layer4(nn.Module):
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.network(x)


class LayerZ(nn.Module):
    
    def __init__(
        self,
        n_obs: int,
    ):
        super().__init__()
        
        self.Z = torch.nn.Parameter(1.0e-8 * torch.ones((n_obs, n_obs)))
        
    def forward(self, x):
        torch.diagonal(self.Z.data).fill_(0)
        self.Z.data = torch.abs(nn.LeakyReLU()(self.Z))
        self.Z.data[self.Z < 1e-5] = 0
        return torch.matmul(self.Z, x)
    
    def getZ(self):
        torch.diagonal(self.Z.data).fill_(0)
        self.Z.data = torch.abs(nn.LeakyReLU()(self.Z))
        self.Z.data[self.Z < 1e-5] = 0
        return self.Z.detach().cpu().numpy()


class LayerS(nn.Module):
    
    def __init__(
        self,
        n_obs_1: int,
        n_obs_2: int,
    ):
        super().__init__()
        
        self.S = torch.nn.Parameter(1.0e-8 * torch.ones((n_obs_1, n_obs_2)))
        
    def forward(self, x):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return torch.matmul(self.S, x)
    
    def getS(self):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return self.S.detach().cpu().numpy()


class SpaDA(nn.Module):
    
    def __init__(
        self,
        n_input: int,
        n_obs_1: int,
        n_obs_2: int,
        n_covar: int,
        n_hidden: int,
        n_latent: int,
        dropout_rate: float = 0.1,
        var_activation: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.n_covar = n_covar
        
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        
        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )
        
        self.layer3 = Layer3(
            n_in=n_latent+n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )
        
        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )
        
        self.layerZ1 = LayerZ(
            n_obs=n_obs_1,
        )
        
        self.layerZ2 = LayerZ(
            n_obs=n_obs_2,
        )
        
        self.layerS = LayerS(
            n_obs_1=n_obs_1,
            n_obs_2=n_obs_2,
        )
        
    def inference(self, x):
        
        x1 = self.layer1(x)
        qz, z = self.layer2(x1)
        
        return dict(x1=x1, z=z, qz=qz)
    
    def generative(self, z, covar):
        
        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))
        
        x4 = self.layer4(x3)
        
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        return dict(x3=x3, x4=x4, pz=pz)
    
    def loss(
        self,
        x_1,
        x_2,
        inference_outputs_1,
        generative_outputs_1,
        inference_outputs_2,
        generative_outputs_2,
        Z_weight,
    ):
        
        kl_divergence_z_1 = kl(inference_outputs_1['qz'], generative_outputs_1['pz']).sum(dim=1)
        kl_divergence_z_2 = kl(inference_outputs_2['qz'], generative_outputs_2['pz']).sum(dim=1)
        reconst_loss_1 = torch.norm(x_1 - generative_outputs_1['x4'])
        reconst_loss_2 = torch.norm(x_2 - generative_outputs_2['x4'])
        loss = (6 - 5 * Z_weight) * (torch.mean(kl_divergence_z_1) + reconst_loss_1 + torch.mean(kl_divergence_z_2) + reconst_loss_2)
        if Z_weight > 0.5:
            loss += Z_weight * torch.norm(inference_outputs_1['x1'] - self.layerZ1(inference_outputs_1['x1']))
            loss += Z_weight * 3 * torch.norm(inference_outputs_1['qz'].loc - self.layerZ1(inference_outputs_1['qz'].loc))
            loss += Z_weight * torch.norm(inference_outputs_2['x1'] - self.layerZ2(inference_outputs_2['x1']))
            loss += Z_weight * 3 * torch.norm(inference_outputs_2['qz'].loc - self.layerZ2(inference_outputs_2['qz'].loc))
            loss += Z_weight * 3 * torch.norm(inference_outputs_1['x1'] - self.layerS(inference_outputs_2['x1']))
            loss += Z_weight * torch.norm(inference_outputs_1['qz'].loc - self.layerS(inference_outputs_2['qz'].loc))
            
        return loss



















