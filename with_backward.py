import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function

# forward
def bp_forward(c_label, c_out):
    c_bar_label = 1 - c_label # false label
    dim = len(c_label) # dimension of labels
    # scalar
    y_i = sum(c_label) # true label numbers
    y_bar_i = dim - y_i # false label numbers

    # fist build Q*Q pairwise subtraction matrix A of c_out
    A = c_out.reshape(len(c_out), 1) - c_out
    # next build Q*Q pairwise logical_and matrix B of c_label (mask)
    B = c_label.reshape(len(c_label), 1) * c_bar_label
    # mask 
    M = np.exp(-A) * B
    # summation and normalization
    Err = np.sum(M) / (y_i * y_bar_i) 

    return Err

# backward 
def bp_backward(c_label, c_out):
    dim = len(c_label) # dimension of labels 6
    # scalar
    y_i = sum(c_label) # true label numbers 3
    y_bar_i = dim - y_i # false label numbers 3
    y_i_id = np.where(c_label==1)[0] # 2 4 5
    y_i_bar_id = np.where(c_label==0)[0] # 0 1 3
    c_in = c_out.take(y_i_id) # [0.9, 0.6, 0.9]
    c_bar = c_out.take(y_i_bar_id) # [0.5, 0.2, 0.3]

    # return grad
    grad = np.empty(dim)
    for id in y_i_id:
        print(id)
        temp = c_out[id] - c_bar # Cj-Cl
        loss = np.sum(np.exp(-temp))
        print(loss)
        grad[id] = - loss / (y_i * y_bar_i)

    for id in y_i_bar_id:
        temp = c_in - c_out[id]
        loss = np.sum(np.exp(-temp))
        grad[id] = loss / (y_i * y_bar_i)

    return grad

class BPmll(Module):
    def __init__(self, c_out):
        super(BPmll, self).__init__()
        # learnable parameters
        self.c_out = Parameter(c_out)

    def forward(self, c_label):
        return BPmllFunction(Function).apply(c_label, self.c_out)


class BPmllFunction(Function):
    @staticmethod
    def forward(ctx, c_label, c_out):
        # detach to cast to NumPy
        c_label, c_out = c_label.detach(), c_out.detach()
        result = bp_forward(c_label, c_out)
        ctx.save_for_backward(c_label, c_out)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx):
        c_label, c_out = ctx.saved_tensors
        grad_out = bp_backward(c_label, c_out)

        return torch.from_numpy(grad_out)  
