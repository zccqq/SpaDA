# -*- coding: utf-8 -*-

from typing import Optional, Tuple
from anndata import AnnData

import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from tqdm import trange

from module import SpaDA
from model_utils import one_hot


def _run_SpaDA(
    data_X_1: np.ndarray,
    data_X_2: np.ndarray,
    covar_1: np.ndarray,
    covar_2: np.ndarray,
    n_epochs: int,
    n_hidden: int,
    n_latent: int,
    device: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    
    if device is None or device == 'cuda':
        if torch.cuda.is_available():
          device = 'cuda'
        else:
          device = 'cpu'
    
    device = torch.device(device)
    
    data_X_1 = torch.Tensor(data_X_1).to(device)
    data_X_2 = torch.Tensor(data_X_2).to(device)
    
    covar_1 = torch.Tensor(covar_1).to(device)
    covar_2 = torch.Tensor(covar_2).to(device)
    
    model = SpaDA(
        n_input=data_X_1.shape[1],
        n_obs_1=data_X_1.shape[0],
        n_obs_2=data_X_2.shape[0],
        n_covar=2,
        n_hidden=n_hidden,
        n_latent=n_latent
    ).to(device)
    
    model.train(mode=True)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
    
    pbar = trange(n_epochs)
    
    for epoch in pbar:
        
        optimizer.zero_grad()
        
        inference_outputs_1 = model.inference(data_X_1)
        generative_outputs_1 = model.generative(inference_outputs_1['z'], covar_1)
        
        inference_outputs_2 = model.inference(data_X_2)
        generative_outputs_2 = model.generative(inference_outputs_2['z'], covar_2)
        
        loss = model.loss(
            data_X_1,
            data_X_2,
            inference_outputs_1,
            generative_outputs_1,
            inference_outputs_2,
            generative_outputs_2,
            epoch/n_epochs
        )
        
        pbar.set_postfix_str(f'loss: {loss.item():.3e}')
        
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        S = model.layerS.getS()
        Z1 = model.layerZ1.getZ()
        Z2 = model.layerZ2.getZ()
        
    return S, Z1, Z2


def run_SpaDA(
    adata: AnnData,
    n_epochs: int = 1000,
    n_hidden: int = 128,
    n_latent: int = 10,
    device: Optional[str] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    
    adata = adata.copy() if copy else adata
    
    batch_info = pd.Categorical(adata.obs['batch'])
    n_batch = batch_info.categories.shape[0]
    batch_index = batch_info.codes.copy()
    batch_index = one_hot(torch.Tensor(batch_index).to(device), n_batch).cpu().numpy()
    
    adata.obsm['batch_index'] = batch_index
    
    adata1 = adata[adata.obs['batch']=='0', :]
    adata2 = adata[adata.obs['batch']=='1', :]
    
    S, Z1, Z2 = _run_SpaDA(
        data_X_1=adata1.X.toarray() if issparse(adata1.X) else adata1.X,
        data_X_2=adata2.X.toarray() if issparse(adata2.X) else adata2.X,
        covar_1=adata1.obsm['batch_index'],
        covar_2=adata2.obsm['batch_index'],
        n_epochs=n_epochs,
        n_hidden=n_hidden,
        n_latent=n_latent,
        device=device,
    )
    
    return S, Z1, Z2



















