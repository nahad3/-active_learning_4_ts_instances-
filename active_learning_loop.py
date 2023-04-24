import sklearn.metrics
import torch
import numpy as np
import wandb
from collections import OrderedDict
from utils_vis import plot_ts_reps
from torch.autograd import grad
from sklearn.metrics import confusion_matrix
from torch import nn
from loss_ssl_tcn import hierarchical_contrastive_loss,centroid_contrast_loss
#from utils.utils import get_two_views
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
'code for psuedolabels and teacher student loop DA'
from sklearn.cluster import KMeans
import collections
from numpy import linalg as LA

def find_nearest(array, value):
    'takes argmin l1 norm for array - value (closest l1 norm index to kmeans centeroid)'
    array = np.asarray(array)
    idx = (LA.norm((array - value),axis=1)).argmin()
    return idx

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



def get_balanced_samples(dataloader,no_classes,no_points):

    'goes through all batches and gets balanced number of samples for each class across each batch. Returns list (no batches x (no_points/no_batches)'

    btch2list = list(dataloader)
    no_batches = len(btch2list)
    no_points_p_class = int(no_points /(no_batches * no_classes))
    out_list = np.zeros((no_batches,int(no_points/no_batches)))
    for k in range(0,no_batches):
        y = btch2list[k]['y_src'].reshape(-1,)

        for i in range(0,no_classes):
            '''
            if k == 0 and i == 0:
                out_list[k, i * (no_points_p_class):(i + 1) * (no_points_p_class)] = np.arange(1700, 1720, 10)
            if k == 0 and i == 1:
                out_list[k,i*(no_points_p_class):(i+1)*(no_points_p_class)] =  np.asarray([475,480])
            elif k == 1 and i == 1:
                out_list[k, i * (no_points_p_class):(i + 1) * (no_points_p_class)] = np.asarray([485, 493])
            elif k == 0 and i == 2:
                out_list[k, i * (no_points_p_class):(i + 1) * (no_points_p_class)] = np.arange(1000, 1020, 10)
            elif k == 0 and i == 3:
                out_list[k, i * (no_points_p_class):(i + 1) * (no_points_p_class)] = np.arange(200, 220, 10)

            else:
            '''
            y_idx = np.where(y == i)[0]
            np.random.shuffle(y_idx)
            #y_idx = np.random.shuffle(np.where(y == i)[0])[0:no_points_p_class]
            out_list[k,i*(no_points_p_class):(i+1)*(no_points_p_class)] = y_idx[0:no_points_p_class]

    return out_list
def train_active_loop(model_feats,model_clfr,train_dataloader,args,device,wanb_con,val_dataload,total_budget=None,
                      queries_per_round = None,init_pool=None,val_each_epoch=True):
    '''Loop for actively training. Starts off with a pool of samples.
    Then completely train (all epochs) using this pool.
    Once traning is done, get more queries through different methods.
    Add queries to the pool and retrain completely.
    Then repeat getting queries and then repeat the process until total queries hit a certain number

    Query selection stragety is selected by args.query_type: 0 for random. 1 for max entropy
    '''

    'Arugments:'
    'Model_feats: Model of learning representations'
    'Model_cflr: Model of learning linear classifier ontop of representations' \
    'device: gpu no'
    'wandb_con: weights and biases object'
    'total budget: Total budget if given, otherwise taken from args'
    'queries_per_round: Samples to be acquired after each round if given., otherwise taken from args' \
    'init_pool: Initial pool to train the modle if given, otherwise generated in a balenced sampled way'
    'val_each_epoch:Validate each epoch result or not. Can be turned off to speed up Experiments for multiple active lrn rounds'
    if total_budget is None: 
        total_budget = args.total_budget
    if queries_per_round is None:
        queries_per_round = args.no_queries

    'randomly train with budget'
    f1_score_list = []
    acc_score_list = []

    optim = torch.optim.Adam(list(model_feats.parameters()) + list(model_clfr.parameters()), args.lr_src,
                             weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.1)
    torch.save({
        'epoch': 0,
        'model_state_dict': model_feats.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }, args.file_path_save_src_feats)
    torch.save({
        'epoch': 0,
        'model_state_dict': model_clfr.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }, args.file_path_save_src_clfr)
    criterion_clfr = torch.nn.CrossEntropyLoss()

    b_val_loss = 1e5
    'need to add ssl loss'
    no_batches  = len(train_dataloader)

    no_labels_p_batch = len(list(train_dataloader)[0]['y_src'].reshape(-1,))

    #start with random number of initial samples (equivalent ot the number of queries)

    '''rand_intial_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=queries_per_round)
    pool= rand_intial_samples.reshape(no_batches,-1)'''

    #get balanced samples for pool if no initial pool provided
    if init_pool is None:
        pool= get_balanced_samples(train_dataloader, 5, no_points=queries_per_round)
        train_round = 1
    else:
        #use provided pool
        pool = np.copy(init_pool)
        train_round = 0
    numb_rounds = int(total_budget/(queries_per_round))

    wandb.define_metric("queries step")
    wandb.define_metric("Active/*", step_metric="queries step")

    for k in range(0,numb_rounds):
        loss_array_train = []
        loss_array_val = []

        #load model at beginning of each round to train from scratch
        checkpoint = torch.load(args.file_path_save_src_feats)
        model_feats.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(args.file_path_save_src_clfr)
        model_clfr.load_state_dict(checkpoint['model_state_dict'])
        params = OrderedDict(model_feats.named_parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.1)
        if train_round  == 1:
            "skip first round of train if initial pool provided"

            #optim = torch.optim.Adam(list(model_feats.parameters()) + list(model_clfr.parameters()), args.lr_src,
            #                         weight_decay=1e-4)



            print(f"Training with no points in pool {pool.reshape(-1, 1).shape[0]}")
            for ep in range(0, args.no_epochs):
                model_feats.train()
                model_clfr.train()
                btch = 0

                for data in train_dataloader:
                    optim.zero_grad()
                    x_src = data['x_src'][:,:,:].to(device)
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



                "Get evaluation for the round"
                model_feats.eval()
                model_clfr.eval()

                if val_each_epoch:
                    #Evaluation done for each Epoch
                    for data in val_dataload:
                        'val dataload batch size normally large enough so that only 1 batch used'
                        x_src = data['x_src'][:,:,:].to(device)
                        y_all = data['y_src'].to(device)
                        no_labels_p_batch = len(y_all)
                        rand_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=int(queries_per_round/1))
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
                            y_true = y_all.reshape(-1,).cpu().numpy().astype(int)
                            y_predicted = torch.argmax(src_lbl_clfr_reshaped,axis=1)
                            f1_score = sklearn.metrics.f1_score(y_true ,y_predicted.cpu().numpy(),average = 'macro')
                            acc = sklearn.metrics.accuracy_score(y_true ,y_predicted.cpu().numpy())
                            wanb_con.log({f"F1 score All Val train {k}": f1_score})
                            wanb_con.log({f"Acc score All Val train {k}": acc})
                            wanb_con.log({f"conf_matrix_all_val": wanb_con.plot.confusion_matrix(y_true = y_true ,
                                                                                               preds = y_predicted.cpu().numpy())})
                            print(f"F1 score {f1_score} at epoch {ep}")
                            print(f"Accuracy {acc} at epoch {ep}")
                            if np.mean(loss_array_val) <= b_val_loss:
                                '''
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
                                '''
                                b_val_loss = np.mean(loss_array_val)

                            wanb_con.log({f"Loss train {k}": np.mean(loss_array_train)})
                            wanb_con.log({f"Loss val: {k}": np.mean(loss_array_val)})
                            active_log_dict_f1 = {"Active/Val F1 score": f1_score, "Active/Val Acc score ": acc,
                                                  "queries step": np.shape(pool.reshape(-1, ))[0]}
                            wanb_con.log(active_log_dict_f1)
                    scheduler.step()




        "Evaluation after training is done"
        for data in val_dataload:
            'val dataload batch size normally large enough so that only 1 batch used'
            x_src = data['x_src'][:, :, :].to(device)
            y_all = data['y_src'].to(device)
            no_labels_p_batch = len(y_all)
            rand_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=int(queries_per_round / 1))

            x_src_feats = model_feats(x_src, params)
            src_lbl_clfr = model_clfr(x_src_feats)
            src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
            y_true = y_all.reshape(-1, ).cpu().numpy().astype(int)
            y_predicted = torch.argmax(src_lbl_clfr_reshaped, axis=1)
            f1_score = sklearn.metrics.f1_score(y_true, y_predicted.cpu().numpy(), average='macro')
            acc = sklearn.metrics.accuracy_score(y_true, y_predicted.cpu().numpy())
            print(f"F1 score at the end of round {k} : {f1_score}")
            print(f"Acc score at the end of round {k} : {acc}")

        #accumulates result in a list
        f1_score_list.append(f1_score)
        acc_score_list.append(acc)

        " Do active aquisition to increase pool if number rounds >1 "
        if numb_rounds > 1:

            model_feats.eval()
            model_clfr.eval()
            pool_sample_add = np.zeros((no_batches,int(queries_per_round/no_batches)))
            btch = 0
            for data in train_dataloader:
                optim.zero_grad()
                x_src = data['x_src'][:,:,:].to(device)


                x_src_feats = model_feats(x_src, params)
                # x_src_feats_reshaped = x_src_feats.reshape(-1,x_src_feats.shape[-1])
                # x_src_feats_sampled = x_src_feats_reshaped[0:rand_samples]
                src_lbl_clfr = model_clfr(x_src_feats)
                src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
                if args.query_type == 1:
                    probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1)
                    entropy = torch.distributions.Categorical(probs=probs).entropy()
                    #get queries per batch such that the total number of queries acorss all batches equals given no of no_queires
                    top_k_ent_indices = torch.topk(entropy, int(queries_per_round/no_batches))[1].cpu().numpy()
                elif args.query_type == 0:
                    top_k_ent_indices = np.random.choice(np.arange(0, src_lbl_clfr_reshaped.shape[0]),
                                                size=int(queries_per_round/no_batches))
                elif args.query_type == 2:
                    'for my entropy freuqency pooling method'
                    probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1)
                    entropy = torch.distributions.Categorical(probs=probs).entropy()
                    entropy = entropy.detach().cpu().numpy()
                    entrop_filt = np.where(entropy > 0.5)[0]
                    x_src_feats_reshaped = x_src_feats.detach().cpu().numpy().reshape(-1, x_src_feats.shape[-1])
                    x_src_feats_filtr = x_src_feats_reshaped[entrop_filt, :]
                    kmeans = KMeans(n_clusters=40, random_state=0).fit(x_src_feats_filtr)
                    pred_clusters = kmeans.predict(x_src_feats_filtr)
                    frequency = collections.Counter(pred_clusters)
                    # get the maximum or top freuqncy points and sample
                    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    top_10_centroids = [val[0] for val in sorted_freq[0:int(queries_per_round/no_batches)]]
                    top_10_kmeans = kmeans.cluster_centers_[top_10_centroids, :]
                    # pool_sample_add =  [val[0] for val in sorted_freq[0:3]]
                    nearest_indices = [find_nearest(x_src_feats_reshaped, top_10_kmeans[value, :]) for value in range(0, int(queries_per_round/no_batches))]
                    top_k_ent_indices = np.asarray(nearest_indices )
                # get new queries for batch
                pool_sample_add[btch, :] = top_k_ent_indices[0:int(queries_per_round/no_batches)]
                btch = btch + 1

            'Add acquired samples to pool'
            pool = np.concatenate((pool, pool_sample_add), axis=1)


        visualize = 0
        # plot representations and entropy after training round
        if visualize == 1:
            i = 0
            list_train_loader = list(train_dataloader)
            x = list_train_loader[0]['x_src'][i]
            y = list_train_loader[0]['y_src'][i]
            fig = plot_ts_reps(model_feats=model_feats, model_clfr=model_clfr, x=x.numpy(), y=y.numpy(), device=device,
                               window=-1, title='str(i)')
            wanb_con.log({f"plot {i} after {k} ": fig})
            print("here")
        # Add queries to batch and then train again


        #set the next round to train if it was set no to train when given initial pool
        train_round = 1
    'Returns f1 score, accuracy score and pool of aquired samples'
    return f1_score_list,acc_score_list,pool








