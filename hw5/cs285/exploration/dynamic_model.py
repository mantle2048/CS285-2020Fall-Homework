import torch.optim as optim
import torch
import numpy as np

from torch import nn
from typing import Union
from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel

class DynamicModel(nn.Module, BaseExplorationModel):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.n_layers = hparams['dyn_n_layers']
        self.size = hparams['dyn_size']
        self.optimizer_spec = optimizer_spec

        self.dynamic_model = ptu.build_mlp(self.ob_dim + 1, self.ob_dim, self.n_layers, self.size)
        self.optimizer = self.optimizer_spec.constructor(
                self.dynamic_model.parameters(),
                **self.optimizer_spec.optim_kwargs
                )

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.dynamic_model.to(ptu.device)

    def forward(
        self,
        ob_no: torch.Tensor,
        ac_na: torch.Tensor,
        next_ob_no: torch.Tensor
    ):

        ob_ac_n = torch.cat((ob_no, ac_na), dim=1)
        pred_next_ob_no = self.dynamic_model(ob_ac_n)

        return torch.norm(pred_next_ob_no - next_ob_no, dim=1)

    def forward_np(
        self,
        ob_no: np.ndarray,
        ac_na: np.ndarray,
        next_ob_no: np.ndarray
    ):

        ob_no      = ptu.from_numpy(ob_no)
        ac_na      = ptu.from_numpy(ac_na).unsqueeze(dim=1)
        next_ob_no = ptu.from_numpy(next_ob_no)

        error = self(ob_no, ac_na, next_ob_no)
        return ptu.to_numpy(error)


    def update(
        self,
        ob_no: np.ndarray,
        ac_na: np.ndarray,
        next_ob_no: np.ndarray
    ):

        ob_no = ptu.from_numpy(ob_no)
        ac_na      = ptu.from_numpy(ac_na).unsqueeze(dim=1)
        next_ob_no = ptu.from_numpy(next_ob_no)

        loss = self.forward(ob_no, ac_na, next_ob_no).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()













