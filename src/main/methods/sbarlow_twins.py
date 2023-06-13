import argparse
from typing import Any, List, Sequence

import torch
import torch.nn as nn
from solo.losses.sbarlow import sbarlow_loss_func
from solo.methods.base import BaseMethod


class SBarlowTwins(BaseMethod):
    def __init__(
        self,
        proj_hidden_dim: int,
        proj_output_dim: int,
        lamb: float,
        scale_loss: float,
        exponent: int,
        group_size: int,
        rand_type: str,
        **kwargs
    ):
        """Barlow-Twins style model (https://arxiv.org/abs/2301.01569)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
            exponent (int): hyperparameter q in the paper (Eqs (6) and (13)).
            group_size (int): block size b in the paper.
            rand_type (str): feature permutation.
        """

        super().__init__(**kwargs)

        self.lamb = lamb
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
        parent_parser = super(SBarlowTwins, SBarlowTwins).add_model_specific_args(
            parent_parser
        )
        parser = parent_parser.add_argument_group("sbarlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)

        parser.add_argument("--exponent", choices=[1, 2], default=2, type=int)
        parser.add_argument("--group_size", default=2048, type=int)
        parser.add_argument("--rand_type", choices=["batch", "none"], default="batch")

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def on_train_batch_start(self, batch, batch_idx):
        if self.rand_type == "batch":
            self.rand_idx = torch.randperm(self.proj_output_dim)

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- barlow twins loss -------
        sbarlow_loss = sbarlow_loss_func(
            z1,
            z2,
            lamb=self.lamb,
            scale_loss=self.scale_loss,
            exponent=self.exponent,
            group_size=self.group_size,
            rand_idx=self.rand_idx,
        )

        self.log("train_sbarlow_loss", sbarlow_loss, on_epoch=True, sync_dist=True)

        return sbarlow_loss + class_loss
