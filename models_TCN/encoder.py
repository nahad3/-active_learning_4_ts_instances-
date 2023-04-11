import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from modules import get_child_dict, Linear
from collections import OrderedDict
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=11, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, params, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        # params here
        #x= x.unsqueeze(0).float()
        fc_weights = get_child_dict(params,'input_fc')

        x = self.input_fc(x,fc_weights)  # B x T x Ch
        #x = x.transpose(1, 2)
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        # params here
        DilatedConvParams = get_child_dict(params,'feature_extractor.net')

        x = self.repr_dropout(self.feature_extractor(x,DilatedConvParams))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

    def all_layer_TCN_out(self, x):
        'gets outputs from intermediate layers (with differnet receptive fields. Consists of a list'
        x = self.input_fc(x)
        x = x.transpose(1, 2)
        all_outputs = self.feature_extractor.get_output_from_each_layer(x)
        return all_outputs

class TSEncoder_wo_mask(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=11, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, params, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        # params here
        #x= x.unsqueeze(0).float()
        fc_weights = get_child_dict(params,'input_fc')

        x = self.input_fc(x,fc_weights)  # B x T x Ch
        #x = x.transpose(1, 2)
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        # params here
        DilatedConvParams = get_child_dict(params,'feature_extractor.net')

        x = self.repr_dropout(self.feature_extractor(x,DilatedConvParams))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x

    def all_layer_TCN_out(self, x):
        'gets outputs from intermediate layers (with differnet receptive fields. Consists of a list'
        x = self.input_fc(x)
        x = x.transpose(1, 2)
        all_outputs = self.feature_extractor.get_output_from_each_layer(x)
        return all_outputs
class TSEnc_with_Proj(nn.Module):
    "Backbone TSencoder with projector"

    def __init__(self, input_dims, enc_out_dims, project_dims, hidden_dims=64, depth=11, mask_mode='binomial'):
        super().__init__()
        self.backbone = TSEncoder( input_dims, enc_out_dims, hidden_dims= hidden_dims, depth=depth, mask_mode=mask_mode)
        self.projector_head = Linear(enc_out_dims, project_dims)

    def forward(self,x,params):
        enc_weights = get_child_dict(params, 'backbone')
        encodings = self.backbone(x,enc_weights)
        proj_weights = get_child_dict(params,'projector_head')
        #encoddintorch.nn.functional.normalize(encodings,dim = 2)
        output = self.projector_head(encodings, proj_weights)
        #output = nn.functional.normalize(output, dim=2)
        return output

    def get_backbone(self,x,params):
        enc_weights = get_child_dict(params, 'backbone')
        encodings = self.backbone(x, enc_weights)
        return encodings
class Projector():
    'Projector module trainable for inner loop'
    def __init__(self, input_dims, output_dims):
        self.projector = Linear(input_dims, output_dims)
    def forward(self,x,params):
        proj_weights = get_child_dict(params, 'projector')
        x = self.input_fc(x, proj_weights)
        return x


class Invariant_feats(nn.Module):
        def __init__(self,input_dim,hidden_dim,out_dim,ssl_feats = 0):
            '''ssl_feats:1 just using SSL encoded featres as input and use MLP projector on top
            ssl_feats:0 : Use a TCM encoder for learning invariant fetures
            '''
            super().__init__()
            self.ssl_feats = ssl_feats
            if self.ssl_feats:

                self.linear_1 = nn.Linear(input_dim, hidden_dim)
                self.relu1 = nn.ReLU()
                self.linear_3 = nn.Linear(hidden_dim, out_dim)
                self.model =nn.Sequential(self.linear_1,self.relu1,self.linear_3)
            else:
                self.model = TSEncoder_wo_mask(input_dims=input_dim,output_dims=out_dim,hidden_dims=hidden_dim,depth=5)
        def forward(self,x):
            if self.ssl_feats:
                return  self.model(x)
            else:
                params =  OrderedDict(self.model.named_parameters())
                return self.model(x,params)
