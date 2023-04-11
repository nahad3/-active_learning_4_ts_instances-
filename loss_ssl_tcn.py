import torch
from torch import nn
import torch.nn.functional as F
import  numpy as np

def hierarchical_contrastive_loss(z1, z2, alpha= 1, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
                #loss += 0 * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)  # along the time axes
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def decomposed_style(z1,z2,alpha=1):
    'print here'
    'style transfer'
    print

def decomposed_content(z1,z2,alpha=1):
    'deconmposed style '

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    #here is where contrast needs to be done with all others
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss





def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def centroid_contrast_loss(cent_s,cent_t,temp):
    'loss to align centroids from souuce and target (CLAD paper: NeurIPS 2021'

    cent_s_norm = nn.functional.normalize(cent_s)
    cent_t_norm = nn.functional.normalize(cent_t)
    #cent_t_rep = cent_t_norm.repeat(cent_s.shape[0],1)

    cent_s_t_mtrx = torch.exp(torch.matmul(cent_s_norm,cent_t_norm.T)/temp) #(5 by 5 matrix)
    cent_s_s_mtrx = torch.exp(torch.matmul(cent_s_norm, cent_s_norm.T)/temp)
    cent_t_t_mtrx = torch.exp(torch.matmul(cent_t_norm,cent_t_norm.T)/temp)
    poss_across_s_t = cent_s_t_mtrx.clone().diagonal()
    poss_across_s_s = cent_s_s_mtrx.clone().diagonal().zero_()
    poss_across_t_t = cent_t_t_mtrx.clone().diagonal().zero_()
    neg_across_s_t = 0.5*torch.sum(cent_s_t_mtrx)
    neg_across_s_s = 0.5*torch.sum(cent_s_s_mtrx)
    neg_across_t_t = 0.5*torch.sum(cent_t_t_mtrx)

    den = neg_across_t_t + neg_across_s_s + neg_across_s_t
    num = poss_across_s_t #num is a vector
    return torch.mean(-torch.log(num/(den+1e-5)))

