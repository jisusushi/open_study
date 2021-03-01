# Defining Loss Function - Masked Mean Squared Error

import torch
import torch.nn as nn

class MSEloss_with_Mask(nn.Module):
    def __init__(self):
        super(MSEloss_with_Mask, self).__init__()

    def forward(self, inputs, targets):
        # Masking into a vector of 1's and 0's.
        mask= (targets!=0)
        mask= mask.float()

        # actual number of ratings
        # Take max to avoid division by zero while caculating loss
        other= torch.Tensor([1.0])
        number_ratings= torch.max(torch.sum(mask), other)
        error= torch.sum(torch.mul(mask, torch.mul((targets-inputs), (targets-inputs))))
        loss= error.div(number_ratings)
        return loss[0]
