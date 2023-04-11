import sklearn.metrics
import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from torch.autograd import grad
from torch import nn
from loss_ssl_tcn import hierarchical_contrastive_loss,centroid_contrast_loss
#from utils.utils import get_two_views
from sklearn.metrics import f1_score

'code for psuedolabels and teacher student loop DA'


def train_soruce(model_feats,model_clfr,train_dataloader,args,device,wanb_con,val_dataload):
    criterion_clfr = torch.nn.CrossEntropyLoss()
    params = OrderedDict(model_feats.named_parameters())
    optim = torch.optim.Adam(list(model_feats.parameters()) + list(model_clfr.parameters()), args.lr_src, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size = 500,gamma=0.1 )
    b_val_loss = 1e5
    'need to add ssl loss'
    for ep in range(0, args.no_epochs_src):
        loss_array_train = []
        loss_array_val = []
        for data in train_dataloader:
            optim.zero_grad()
            x_src = data['x_src'].to(device)
            y_src = data['y_src'].to(device)

            x_src_feats = model_feats(x_src, params)
            src_lbl_clfr = model_clfr(x_src_feats)
            loss_lbl_src = criterion_clfr(src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1]),
                                          y_src.reshape(-1, ).long())
            loss_lbl_src.backward()
            optim.step()
            loss_array_train.append(loss_lbl_src.item())

        model_feats.eval()
        model_clfr.eval()

        for data in val_dataload:
            x_src = data['x_src'].to(device)
            y_src = data['y_src'].to(device)

            x_src_feats = model_feats(x_src, params)
            src_lbl_clfr = model_clfr(x_src_feats)
            loss_lbl_src = criterion_clfr(src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1]),
                                          y_src.reshape(-1, ).long())
            loss_array_val.append(loss_lbl_src.item())
        wanb_con.log({"Loss train: {}": np.mean(loss_array_train)})
        wanb_con.log({"Loss val: {}": np.mean(loss_array_val)})

        if np.mean(loss_array_val) <= b_val_loss:
            torch.save({
               'epoch': ep,
                'model_state_dict': model_feats.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            }, args.file_path_save_src_feats)
            torch.save({
                'epoch': ep,
                'model_state_dict': model_clfr.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            }, args.file_path_save_src_clfr)
            print("Saving")
            print(f"F1 score {f1_score}")
            b_val_loss = np.mean(loss_array_val)
        scheduler.step()




def train_active_loop(model_feats,model_clfr,train_dataloader,args,device,wanb_con,val_dataload):
    '''Loop for actively training. Starts off with a pool of samples.
    Then completely train (all epochs) using this pool.
    Once traning is done, get more queries through different methods.
    Add queries to the pool and retrain completely.
    Then repeat getting queries and then repeat the process until total queries hit a certain number

    Query selection stragety is selected by args.query_type: 0 for random. 1 for max entropy
    '''



    'randomly train with budget'


    criterion_clfr = torch.nn.CrossEntropyLoss()
    params = OrderedDict(model_feats.named_parameters())
    optim = torch.optim.Adam(list(model_feats.parameters()) + list(model_clfr.parameters()), args.lr_src, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size = 500,gamma=0.1 )
    b_val_loss = 1e5
    'need to add ssl loss'
    no_batches  = len(train_dataloader)

    no_labels_p_batch = len(list(train_dataloader)[0]['y_src'].reshape(-1,))

    #start with random number of initial samples (equivalent ot the number of queries)
    rand_intial_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=args.no_queries)


    pool= rand_intial_samples.reshape(no_batches,-1)

    no_loops = int(args.total_budget/(args.no_queries))

    wandb.define_metric("queries step")
    wandb.define_metric("Active/*", step_metric="queries step")

    for k in range(0,no_loops):
        model_feats.train()
        model_clfr.train()
        for ep in range(0, args.no_epochs):
            loss_array_train = []
            loss_array_val = []
            btch = 0

            for data in train_dataloader:

                optim.zero_grad()
                x_src = data['x_src'].to(device)
                y_src = data['y_src'].reshape(-1,)#.to(device)
                x_src_feats = model_feats(x_src, params)

                src_lbl_clfr = model_clfr(x_src_feats)
                src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])

                pool_samples = pool[btch,:]
                y_pool_samples = y_src[pool_samples].to(device)
                src_lbl_clfr_reshped_sampled = src_lbl_clfr_reshaped[pool_samples,:]
                loss_lbl_src = criterion_clfr(src_lbl_clfr_reshped_sampled ,
                                              y_pool_samples)
                loss_lbl_src.backward()
                optim.step()
                loss_array_train.append(loss_lbl_src.item())
                btch = btch + 1

            #Add to batch

            model_feats.eval()
            model_clfr.eval()


            f1_score = 0
            for data in val_dataload:
                'val dataload batch size normally large enough so that only 1 batch used'
                x_src = data['x_src'].to(device)
                y_all = data['y_src'].to(device)
                no_labels_p_batch = len(y_all)
                rand_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=int(args.no_queries/1))
                no_labels_p_batch = len(y_all.reshape(-1,))
                #rand_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=int(budget * no_labels_p_batch))

                y_sampled = y_all.reshape(-1,).to(device)[rand_samples]

                x_src_feats = model_feats(x_src, params)
                src_lbl_clfr = model_clfr(x_src_feats)
                src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
                src_lbl_clfr_reshped_sampled = src_lbl_clfr_reshaped[rand_samples, :]

                loss_lbl_src = criterion_clfr(src_lbl_clfr_reshped_sampled,
                                              y_sampled.long())
                loss_array_val.append(loss_lbl_src.item())
                #F1 score on all points (not only sampled labeled points)
                if ep %100 == 0:
                    y_predicted = torch.argmax(src_lbl_clfr_reshaped,axis=1)
                    f1_score = sklearn.metrics.f1_score(y_all.reshape(-1,).cpu().numpy().astype(int),y_predicted.cpu().numpy(),average = 'macro')
                    acc = sklearn.metrics.accuracy_score(y_all.reshape(-1,).cpu().numpy().astype(int),y_predicted.cpu().numpy())
                    wanb_con.log({f"F1 score All Val train {k}": f1_score})
                    wanb_con.log({f"Acc score All Val train {k}": acc})
                    wanb_con.log({f"conf_matrix_all_val": wanb_con.plot.confusion_matrix(y_true = y_all.reshape(-1,).cpu().numpy().astype(int),
                                                                                       preds = y_predicted.cpu().numpy())})
                    print(f"F1 score {f1_score} at epoch {ep}")
                    print(f"Accuracy {acc} at epoch {ep}")
                    if np.mean(loss_array_val) <= b_val_loss:
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': model_feats.state_dict(),
                            'optimizer_state_dict': optim.state_dict()
                        }, args.file_path_save_src_feats)
                        torch.save({
                            'epoch': ep,
                            'model_state_dict': model_clfr.state_dict(),
                            'optimizer_state_dict': optim.state_dict()
                        }, args.file_path_save_src_clfr)
                        print("Saving")
                        b_val_loss = np.mean(loss_array_val)

            wanb_con.log({f"Loss train {k}": np.mean(loss_array_train)})
            wanb_con.log({f"Loss val: {k}": np.mean(loss_array_val)})
            scheduler.step()
        "Add queries to pool and retrain"
        model_feats.eval()
        model_clfr.eval()

        active_log_dict_f1 = {"Active/Val F1 score": f1_score,  "Active/Val Acc score ":acc,"queries step": np.shape(pool.reshape(-1,))[0]}
        wanb_con.log(active_log_dict_f1)
        pool_sample_add = np.zeros((no_batches,int(args.no_queries/no_batches)))
        btch = 0
        for data in train_dataloader:
            optim.zero_grad()
            x_src = data['x_src'].to(device)


            x_src_feats = model_feats(x_src, params)
            # x_src_feats_reshaped = x_src_feats.reshape(-1,x_src_feats.shape[-1])
            # x_src_feats_sampled = x_src_feats_reshaped[0:rand_samples]
            src_lbl_clfr = model_clfr(x_src_feats)
            src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
            if args.query_type == 1:
                probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1)
                entropy = torch.distributions.Categorical(probs=probs).entropy()
                #get queries per batch such that the total number of queries acorss all batches equals given no of no_queires
                top_k_ent_indices = torch.topk(entropy, int(args.no_queries/no_batches))[1].cpu().numpy()
            elif args.query_type == 0:
                top_k_ent_indices = np.random.choice(np.arange(0, src_lbl_clfr_reshaped.shape[0]),
                                            size=int(args.no_queries/no_batches))

            # get new queries for batch
            pool_sample_add[btch, :] = top_k_ent_indices
            btch = btch + 1

        wanb_con.log({f"Loss val: {k}": np.mean(loss_array_val)})
        # Add queries to batch and then train again
        pool = np.concatenate((pool, pool_sample_add), axis=1)








