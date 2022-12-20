import numpy as np
import torch
from torch.nn.modules.module import Module


class BpmllLoss(Module):
    def __init__(self):
        super(BpmllLoss, self).__init__()

    # c_out is input, c_label is target
    def forward(self, input, target):
        dim = torch.sum(target, 1)
        dim_bar = target.size()[1] - dim
        norm = torch.mul(dim, dim_bar)
        target_bar = 1 + torch.neg(target) # false label
        # fist build Q*Q pairwise subtraction matrix A of c_out
        A = torch.sub(torch.unsqueeze(input, 2), torch.unsqueeze(input, 1))
        # next build Q*Q pairwise logical_and matrix B of c_label (mask)
        B = torch.mul(torch.unsqueeze(target, 2), torch.unsqueeze(target_bar, 1))
        # mask 
        M = torch.exp(torch.neg(A)) * B
        # summation and normalization
        loss_i = torch.div(torch.sum(M, (2,1)), norm)
        loss = torch.sum(loss_i)
        loss.requires_grad_(True)

        return loss
        

# testing
c_label = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]]) # true label: target
c_out = torch.tensor([[0.5, 0.2, 0.9, 0.3, 0.6, 0.9],
                      [0.7, 0.6, 0.8, 0.2, 0.2, 0.9]], requires_grad=True) # network output: input
module = BpmllLoss()
output = module(c_out, c_label)
output.backward()
print(c_out.grad)
