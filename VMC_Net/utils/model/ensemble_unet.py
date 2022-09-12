import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Tuple
import fastmri
from fastmri.data import transforms

from unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x, mean, std = self.norm(x)
        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unnorm(x, mean, std)
        x = x.squeeze(dim=0)
        return x

class EnsembleNet(nn.Module):
    def __init__(self, varnet, multidomainnet, crossdomainnet7, crossdomainnet8,
        chans: int = 8,
        num_pools: int = 4,
        in_chans: int = 4,
        out_chans: int = 1,
        drop_prob: float = 0.0,
    ):
        super(EnsembleNet, self).__init__()

        self.varnet = varnet
        self.multidomainnet = multidomainnet
        self.crossdomainnet7 = crossdomainnet7
        self.crossdomainnet8 = crossdomainnet8
        
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        varnet_output = self.varnet(masked_kspace, mask)
        multidomainnet_output = self.multidomainnet(masked_kspace, mask)
        crossdomainnet7_output = self.crossdomainnet7(masked_kspace, mask)
        crossdomainnet8_output = self.crossdomainnet8(masked_kspace, mask)

        stack = torch.cat((varnet_output, multidomainnet_output, crossdomainnet7_output, crossdomainnet8_output), dim = 0)
        stack = stack.unsqueeze(0)

        output = self.norm_unet(stack)
       
        return output
