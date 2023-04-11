from .encoder  import Invariant_feats
from utils.functions import ReverseLayerF
from torch import nn
import torch
class DA_model(nn.Module):
    'Adapted from https://github.com/fungtion/DANN/blob/master/models/model.py'
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=11,no_classes=6, mask_mode='binomial',
                 use_ssl_feats=0,spectr_mmd=False, ar=0):
        super().__init__()
        self.feature_network = Invariant_feats(input_dims,hidden_dims,output_dims,ssl_feats=use_ssl_feats)
        self.clfr = nn.Sequential((nn.Linear(input_dims,no_classes)))
        #self.clfr = nn.Sequential((nn.Linear(output_dims,10)))
        #self.clfr=  nn.Sequential((nn.Linear(output_dims,20),nn.ReLU(),nn.Linear(20,10))
        if ar == 1:
            self.dom_clfr = Discriminator_AR(input_channels=input_dims, ar_hidden_dim=10, num_layers=2)

        else:
            if spectr_mmd:
                self.dom_clfr = DSKN2(input_channels=input_dims,hidden_dim=128)
            else:
                self.dom_clfr = nn.Sequential((nn.Linear(input_dims, 1)))
    def forward(self,x,alpha):
        feats = self.feature_network(x)
        feats = x
        reverse_feat = ReverseLayerF.apply(feats,alpha)
        class_output = self.clfr(x)

        domain_output = self.dom_clfr(reverse_feat)
        return class_output, domain_output

    def get_invariant_feats(self,x):
        feats = x
        return feats




class classifier_model(nn.Module):
    def __init__(self, input_dims, no_classes):
        super().__init__()
        self.clfr = nn.Sequential((nn.Linear(input_dims,no_classes)))
    def forward(self,x):
        class_output = self.clfr(x)
        return class_output


class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, input_channels,ar_hidden_dim,num_layers,bi_directional = True):
        """Init discriminator."""
        self.input_dim = input_channels
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(input_size=input_channels, hidden_size=ar_hidden_dim,num_layers = num_layers,bidirectional=bi_directional, batch_first=True)
        #self.DC = nn.Linear(configs.disc_AR_hid+configs.disc_AR_hid*configs.disc_AR_bid, 1)
        self.DC = nn.Linear(ar_hidden_dim+ar_hidden_dim,1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        #input = input.view(input.size(0),-1, self.input_dim )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = encoder_outputs[:, -1, :]
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list



class DA_model_Spectral_MMD(nn.Module):
    'Adapted from https://github.com/fungtion/DANN/blob/master/models/model.py'
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=11,no_classes=6, mask_mode='binomial',use_ssl_feats=0):
        super().__init__()
        self.feature_network = Invariant_feats(input_dims,hidden_dims,output_dims,ssl_feats=use_ssl_feats)
        self.clfr = nn.Sequential((nn.Linear(input_dims,no_classes)))
        #self.clfr = nn.Sequential((nn.Linear(output_dims,10)))
        #self.clfr=  nn.Sequential((nn.Linear(output_dims,20),nn.ReLU(),nn.Linear(20,10))
        #self.dom_clfr = nn.Sequential((nn.Linear(input_dims,1)))
        self.dom_clfr = DSKN2(input_channels=input_dims,hidden_dim=128)
    def forward(self,x,alpha):
        #feats = self.feature_network(x)
        feats = x
        reverse_feat = ReverseLayerF.apply(feats,alpha)
        class_output = self.clfr(x)

        domain_output = self.dom_clfr(reverse_feat)
        return class_output, domain_output

    def get_invariant_feats(self,x):
        feats = x
        return feats



class Cosine_activation(nn.Module):
    def __init__(self):
        super().__init__()
        return
    def forward(self, x):
        return torch.cos(x)
class Spectral_map(nn.Module):
    def __init__(self,input_channels = 64, hidden_dim = 10):
        super().__init__()

        self.layer1 =  nn.Linear(input_channels,int(hidden_dim/2))
        self.layer2 = nn.Linear(input_channels, int(hidden_dim/2))
        self.btch_norm = nn.BatchNorm1d(num_features=int(hidden_dim/2))
        self.cos_activ = Cosine_activation()
        self.relu_activ = nn.ReLU()
    def forward(self,x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(x)
        #out_norm1 = self.btch_norm(layer1_out)
        #out_norm2 = self.btch_norm(layer2_out)
        out1 = self.cos_activ(layer1_out)
        out2 = self.relu_activ(layer2_out)
        return torch.cat((out1, out2), axis= -1)


class DSKN2(nn.Module):
    def __init__(self,input_channels = 64, hidden_dim = 10):
        super().__init__()
        self.layer1 = Spectral_map(input_channels=input_channels,hidden_dim=hidden_dim)
        self.layer2 = Spectral_map(input_channels=hidden_dim, hidden_dim=int(hidden_dim/2))

    def forward(self,x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out2


def MMD(x, y, kernel,device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """



    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))


    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
        val = torch.mean(XX + YY - 2. * XY)


    if kernel == 'none':
        XX = xx
        YY = yy
        XY = zz
        val = torch.mean(XX + YY - 2. * XY)

    return val