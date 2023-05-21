import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

from utils.utils import get_two_views
from loss_ssl_tcn import hierarchical_contrastive_loss


def train_self_sup(model_feats, device, train_dataloader, no_epochs,  val_dataloader,
                      wandb, file_path_bbone,args,
                      file_path_save='./saved_models/DA_robo_b_to_c_supervised' ):
    'code for training on supervised source data'
    # clf_model = model_da.to(device)

    params = OrderedDict(model_feats.named_parameters())
    # feature_extractor = model_da.c.to(device)
    # discriminator = model_da.clfr.to(device)

    # critic = model_da.dom_clfr.to(device)


    optim = torch.optim.Adam(list(model_feats.parameters()) , lr=1e-4,
                                 weight_decay=1e-3)

    ssl_criterion = hierarchical_contrastive_loss
    clf_criterion = nn.CrossEntropyLoss()
    b_val_loss = 1e10


    for ep in range(1, no_epochs):
        # batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        train_loss_array = []
        val_loss_array = []


        print("epoch {}".format(ep))

        model_feats.train()
        model_awg = torch.optim.swa_utils.AveragedModel(model_feats)

        'train loop'
        for data in train_dataloader:

            optim.zero_grad()
            x_src = data['x_src']#.to(device)
            x_trgt = data['x_trgt']


            x_src_feats = model_feats(x_src.to(device),params)
            source_x = x_src_feats


            'self supervised losses. Get two views'
            x1_src, x2_src, crop_l = get_two_views(x_src)


            out1_src = model_feats(x1_src.cuda(device), params)
            out2_src = model_feats(x2_src.cuda(device), params)
            out1_src = out1_src[:, -crop_l:]
            out2_src = out2_src[:, :crop_l]

            loss = ssl_criterion( out1_src,out2_src,temporal_unit=0)


            loss.backward()
            optim.step()
            model_awg.update_parameters(model_feats)
            train_loss_array.append(loss.item())


        'logging train loop results'
        mean_train_loss = np.mean(train_loss_array)

        wandb.log(
            {'Loss/train ': mean_train_loss})

        'validation loop'
        model_feats.eval()

        for data in val_dataloader:
            x_src = data['x_src']  # .to(device)




            x_src_feats = model_feats(x_src.to(device),params)

            x1_src, x2_src, crop_l = get_two_views(x_src)


            out1_src = model_feats(x1_src.cuda(device), params)
            out2_src = model_feats(x2_src.cuda(device), params)
            out1_src = out1_src[:, -crop_l:]
            out2_src = out2_src[:, :crop_l]

            loss_val = ssl_criterion(out1_src, out2_src, temporal_unit=0)


            val_loss_array.append(loss_val.item())


        'logging validation results'
        mean_val_loss = np.mean(val_loss_array)

        wandb.log(
            {'Loss/val': mean_val_loss})
        print("val loss {}".format(mean_val_loss))
        if np.abs(mean_val_loss) <= b_val_loss:

            torch.save({
                'epoch': ep,
                'model_state_dict': model_feats.state_dict()
            }, file_path_bbone)
            print("Saving")
            b_val_loss = np.abs(mean_val_loss)