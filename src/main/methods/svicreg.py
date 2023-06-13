import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.svicreg import svicreg_loss_func
from solo.methods.base import BaseMethod


class SVICReg(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        scale_loss: float,
        exponent: int,
        group_size: int,
        rand_type: str,
        **kwargs
    ):
        """VICReg style model (https://arxiv.org/abs/2301.01569)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
            scale_loss (float): scaling factor of the loss.
            exponent (int): hyperparameter q in the paper (Eqs (6) and (13)).
            group_size (int): block size b in the paper.
            rand_type (str): feature permutation.
        """

        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.proj_output_dim = proj_output_dim
        self.exponent = exponent
        self.group_size = group_size
        self.rand_type = rand_type
        self.rand_idx = None

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parent_parser = super(SVICReg, SVICReg).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("svicreg")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)

        # type of proposal
        parser.add_argument("--scale_loss", default=1.0, type=float)
        parser.add_argument("--exponent", choices=[1, 2], default=2, type=int)
        parser.add_argument("--group_size", default=2048, type=int)
        parser.add_argument("--rand_type", choices=["batch", "none"], default="batch")
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the encoder and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def on_train_batch_start(self, batch, batch_idx):
        if self.rand_type == "batch":
            self.rand_idx = torch.randperm(self.proj_output_dim)

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        svicreg_loss = svicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
            scale_loss=self.scale_loss,
            exponent=self.exponent,
            group_size=self.group_size,
            rand_idx=self.rand_idx,
        )

        self.log("train_svicreg_loss", svicreg_loss, on_epoch=True, sync_dist=True)

        return svicreg_loss + class_loss
