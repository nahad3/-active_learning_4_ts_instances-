import torch
import torch.nn.functional as F

def soft_thresh(w,lamb):
    'w is the orginial unsparse vector, lambda is the regularization parameter for the l1 noirm'
    return F.relu(torch.abs(w) - lamb) * torch.sign(w)