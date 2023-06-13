from solo.methods.barlow_twins import BarlowTwins
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.deepclusterv2 import DeepClusterV2
from solo.methods.dino import DINO
from solo.methods.linear import LinearModel
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.mocov3 import MoCoV3
from solo.methods.nnbyol import NNBYOL
from solo.methods.nnclr import NNCLR
from solo.methods.nnsiam import NNSiam
from solo.methods.ressl import ReSSL
from solo.methods.simclr import SimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.supcon import SupCon
from solo.methods.swav import SwAV
from solo.methods.vibcreg import VIbCReg
from solo.methods.vicreg import VICReg
from solo.methods.wmse import WMSE

from solo.methods.svicreg import SVICReg
from solo.methods.sbarlow_twins import SBarlowTwins

METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "mocov3": MoCoV3,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
    "svicreg": SVICReg,
    "sbarlow_twins": SBarlowTwins,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "MoCoV3",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
    "SVICReg",
    "SBarlowTwins",
]
