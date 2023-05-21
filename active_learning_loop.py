import sklearn.metrics
import torch
import numpy as np
import wandb
from collections import OrderedDict
from utils_vis import plot_ts_reps

from tslearn.clustering import TimeSeriesKMeans
import math
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
from torch.autograd import grad
from sklearn.metrics import confusion_matrix
from torch import nn
from loss_ssl_tcn import hierarchical_contrastive_loss,centroid_contrast_loss
#from utils.utils import get_two_views
from sklearn.metrics import f1_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
'code for psuedolabels and teacher student loop DA'
from sklearn.cluster import KMeans
import collections
from numpy import linalg as LA
from kmeans_pytorch import kmeans_fit
from collections import Counter
import math
from utils.slid_wind_entropy import get_windows_4_ent,get_entp_score,get_jumped_indices


np.random.seed(2023)
torch.manual_seed(2023)

def find_nearest(array, value):
    'takes argmin l1 norm for array - value (closest l1 norm index to kmeans centeroid)'
    'Ensures values from existing pool are eliminated'
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


def get_diverse_samples(X,no_samples):
    'takes in a dataset and retursn the most diverse samples from it'
    kmeans = KMeans(n_clusters=no_samples, random_state=0).fit(X)


    nearest_indices = [find_nearest(X, kmeans.cluster_centers_[value,:]) for value in
                       range(0, no_samples)]
    return nearest_indices

def get_balanced_samples(dataloader,no_classes,no_points,select_most_diverse=False):

    'goes through all batches and gets balanced number of samples for each class across each batch. Returns list (no batches x (no_points/no_batches)'
    'select more diverse flag. Does clustering on data for each class to select most diverse samples. Set to False by default'
    btch2list = list(dataloader)
    no_batches = len(btch2list)
    no_points_p_class = int(no_points /(no_batches * no_classes))
    out_list = np.zeros((no_batches,int(no_points/no_batches)))

    for k in range(0,no_batches):
        y = btch2list[k]['y'].reshape(-1,)
        x = btch2list[k]['x'].reshape(-1, btch2list[k]['x'].shape[-1])
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

            if select_most_diverse:
                idx = get_diverse_samples(x[y_idx,:],no_points_p_class)
                out_list[k, i * (no_points_p_class):(i + 1) * (no_points_p_class)] = idx
            else:
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

    no_labels_p_batch = len(list(train_dataloader)[0]['y'].reshape(-1,))

    #start with random number of initial samples (equivalent ot the number of queries)

    '''rand_intial_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=queries_per_round)
    pool= rand_intial_samples.reshape(no_batches,-1)'''

    #get balanced samples for pool if no initial pool provided
    if init_pool is None:
        pool= get_balanced_samples(train_dataloader, 5, no_points=queries_per_round)

        train_round = 1
    elif init_pool =='random':
        rand_intial_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=queries_per_round)
        pool = rand_intial_samples.reshape(no_batches, -1)
        train_round = 1

    else:
        pool = np.copy(init_pool)
        train_round = 0
    numb_rounds = int(total_budget/(queries_per_round))

    wandb.define_metric("queries step")
    wandb.define_metric("Val*", step_metric="queries step")

    for k in range(0,numb_rounds):
        loss_array_train = []
        loss_array_val = []

        #load model at beginning of each round to train from scratch
        checkpoint = torch.load(args.file_path_save_src_feats)
        model_feats.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(args.file_path_save_src_clfr)
        model_clfr.load_state_dict(checkpoint['model_state_dict'])
        params = OrderedDict(model_feats.named_parameters())
        optim = torch.optim.Adam(list(model_feats.parameters()) + list(model_clfr.parameters()), args.lr_src,
                                 weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.1)
        if train_round  == 1:
            "skip first round of train if initial pool provided"



            loss_array_train = []
            loss_array_val = []
            print(f"Training with no points in pool {pool.reshape(-1, 1).shape[0]}")
            for ep in range(0, args.no_epochs):
                model_feats.train()
                model_clfr.train()
                btch = 0

                for data in train_dataloader:
                    optim.zero_grad()
                    x_src = data['x'][:,:,:].to(device)
                    y_src = data['y'].reshape(-1,)#.to(device)
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
                        x_src = data['x'][:,:,:].to(device)
                        y_all = data['y'].to(device)
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
            x_src = data['x'][:, :, :].to(device)
            y_all = data['y'].to(device)
            no_labels_p_batch = len(y_all)
            rand_samples = np.random.choice(np.arange(0, no_labels_p_batch), size=int(queries_per_round / 1))

            x_src_feats = model_feats(x_src, params)
            src_lbl_clfr = model_clfr(x_src_feats)
            src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
            y_true = y_all.reshape(-1, ).cpu().numpy().astype(int)
            y_predicted = torch.argmax(src_lbl_clfr_reshaped, axis=1)
            f1_score = sklearn.metrics.f1_score(y_true, y_predicted.cpu().numpy(), average='macro')
            acc = sklearn.metrics.accuracy_score(y_true, y_predicted.cpu().numpy())
            print(f"F1 score at the end of round {k}: {f1_score}")
            print(f"Acc score at the end of round {k} : {acc}")
            if numb_rounds > 1:
                active_log_dict_f1 = {f"Val F1 score/{args.dict_type[args.query_type]}/  for run number {args.run_no} ": f1_score,
                                      f"Val ACC score/{args.dict_type[args.query_type]}/  for run number {args.run_no} ": acc,
                                      "queries step": np.shape(pool.reshape(-1, ))[0]}

                wanb_con.log(active_log_dict_f1)
        #accumulates result in a list
        f1_score_list.append(f1_score)
        acc_score_list.append(acc)

        " Do active aquisition to increase pool if number rounds >1 "
        if numb_rounds > 1:

            model_feats.eval()
            model_clfr.eval()
            pool_sample_add = np.zeros((no_batches,int(queries_per_round/no_batches)))
            n_samples = int(queries_per_round/no_batches)
            btch = 0
            for data in train_dataloader:
                optim.zero_grad()

                # x_src: n * 1500 * 12
                x_src = data['x'][:,:,:].to(device)
                y_src = data['y']
                # x_srz_feats: n * 1500 * |params|
                x_src_feats = model_feats(x_src, params)
                x_src_feats_reshaped = x_src_feats.reshape(-1, x_src_feats.shape[-1])

                # x_src_feats_reshaped = x_src_feats.reshape(-1,x_src_feats.shape[-1])
                # x_src_feats_sampled = x_src_feats_reshaped[0:rand_samples]
                src_lbl_clfr = model_clfr(x_src_feats)

                # src_lbl_clfr_reshaped: (n*1500) * |params|
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
                    if args.args.entropy_prcntile is not None:
                        args.entropy_thresh = np.percentile(entropy,args.entropy_prcntile)
                    entrop_filt = np.where(entropy > args.entropy_thresh)[0]


                    x_src_feats_reshaped = x_src_feats.detach().cpu().numpy().reshape(-1, x_src_feats.shape[-1])
                    x_src_feats_filtr = x_src_feats_reshaped[entrop_filt, :]


                    kmeans = KMeans(n_clusters=args.no_clusters, random_state=0).fit(x_src_feats_filtr)


                    #kmeans = TimeSeriesKMeans(n_clusters=10,
                    #                          n_init=2,
                    #                          metric="dtw",
                    #                          verbose=True,
                    #                          max_iter_barycenter=10).fit( x_src_feats_filtr)
                    pred_clusters = kmeans.predict(x_src_feats_filtr)
                    frequency = collections.Counter(pred_clusters)


                    "Get the maximum or top freuqncy points and sample"
                    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    top_10_centroids = [val[0] for val in sorted_freq[0:int(queries_per_round/no_batches)]]
                    top_10_kmeans = kmeans.cluster_centers_[top_10_centroids, :]

                    "Ensure nearest indices are not from existing pool values"
                    x_src_feats_reshaped[pool[btch,:].astype(int),:] = 1e5
                    nearest_indices = [find_nearest(x_src_feats_reshaped, top_10_kmeans[value, :]) for value in range(0, int(queries_per_round/no_batches))]
                    top_k_ent_indices = np.asarray(nearest_indices )
                elif args.query_type == 5:
                    'This method is a mixture of clustering and random sampling'
                    y_reshape = y_src.reshape(-1, )
                    idx_0 = torch.where(y_reshape == 0)[0].cpu().numpy()
                    random_idxs = idx_0




                    'function to get segments or points where entropy is larger than a threshold'
                    windows = get_windows_4_ent(x_src_feats_reshaped[0:6000, :], window=300, stride=1)
                    entp_scores = get_entp_score(windows, bins=40)

                    #seg_list,t_len = get_jumped_indices(y_reshape.clone().cpu(),idx_0)
                    probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1)
                    entropy = torch.distributions.Categorical(probs=probs).entropy()
                    entropy = entropy.detach().cpu().numpy()
                    entrop_filt = np.where(entropy > args.entropy_thresh)[0]
                    #args.entropy_thresh = args.entropy_thresh - 0.10
                    'Removing points where random filtering is to be performed'
                    entrop_filt = np.setdiff1d(entrop_filt,random_idxs)

                    #no_random = math.ceil((len(random_idxs)/len(entp_scores) ) * int(queries_per_round / no_batches))
                    no_random = 4

                    random_samples = np.random.choice(random_idxs,
                                                size=no_random)
                    x_src_feats_reshaped = x_src_feats.detach().cpu().numpy().reshape(-1, x_src_feats.shape[-1])
                    x_src_feats_filtr = x_src_feats_reshaped[entrop_filt, :]

                    kmeans = KMeans(n_clusters=args.no_clusters, random_state=0).fit(x_src_feats_filtr)
                    #kmeans = MeanShift(bandwidth=0.5).fit(x_src_feats_filtr)
                    pred_clusters = kmeans.predict(x_src_feats_filtr)
                    frequency = collections.Counter(pred_clusters)
                    # get the maximum or top freuqncy points and sample
                    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    top_10_centroids = [val[0] for val in sorted_freq[0:int(queries_per_round / no_batches)]]
                    top_10_kmeans = kmeans.cluster_centers_[top_10_centroids, :]
                    # pool_sample_add =  [val[0] for val in sorted_freq[0:3]]

                    #use the rest of the budget for the conventional method
                    nearest_indices = [find_nearest(x_src_feats_reshaped, top_10_kmeans[value, :]) for value in
                                       range(0, (int(queries_per_round / no_batches) - no_random))]
                    top_k_ent_indices = np.asarray(nearest_indices)
                    top_k_ent_indices = np.concatenate((top_k_ent_indices,random_samples),axis=0)
                    #kde = KernelDensity(kernel='gaussian', bandwidth=args.bandwidth).fit(x_src_feats_filtr)
                    #kde_score = kde.score_samples(x_src_feats_filtr)

                    #np.argsort(kde_score)


                    #pred_clusters = kmeans.predict(x_src_feats_filtr)
                    #frequency = collections.Counter(pred_clusters)
                    # get the maximum or top freuqncy points and sample
                    #sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    #top_10_centroids = [val[0] for val in sorted_freq[0:int(queries_per_round/no_batches)]]
                    #top_10_kmeans = kmeans.cluster_centers_[top_10_centroids, :]
                    # pool_sample_add =  [val[0] for val in sorted_freq[0:3]]
                    #nearest_indices = [find_nearest(x_src_feats_reshaped, top_10_kmeans[value, :]) for value in range(0, int(queries_per_round/no_batches))]
                    #top_k_ent_indices = np.asarray(nearest_indices )

                elif args.query_type == 6:
                    'Experimetnal method that uses marginal confidence instead of entropy for uncertainty'
                    probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1)
                    top2_probs = torch.topk(probs, 2, dim=1)[0]

                    marg_confd = 1-(top2_probs[:,0] - top2_probs[:,1])
                    if args.args.marg_confd_prcntile is not None:
                        args.marg_confd_thresh = np.percentile(marg_confd,args.marg_confd_prcntile)
                    marg_confd_filt = np.where(marg_confd > args.marg_confd_thresh)[0]


                    x_src_feats_reshaped = x_src_feats.detach().cpu().numpy().reshape(-1, x_src_feats.shape[-1])
                    x_src_feats_filtr = x_src_feats_reshaped[marg_confd_filt, :]


                    kmeans = KMeans(n_clusters=args.no_clusters, random_state=0).fit(x_src_feats_filtr)


                    #kmeans = TimeSeriesKMeans(n_clusters=10,
                    #                          n_init=2,
                    #                          metric="dtw",
                    #                          verbose=True,
                    #                          max_iter_barycenter=10).fit( x_src_feats_filtr)
                    pred_clusters = kmeans.predict(x_src_feats_filtr)
                    frequency = collections.Counter(pred_clusters)


                    "Get the maximum or top freuqncy points and sample"
                    sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
                    top_10_centroids = [val[0] for val in sorted_freq[0:int(queries_per_round/no_batches)]]
                    top_10_kmeans = kmeans.cluster_centers_[top_10_centroids, :]

                    "Ensure nearest indices are not from existing pool values"
                    x_src_feats_reshaped[pool[btch,:].astype(int),:] = 1e5
                    nearest_indices = [find_nearest(x_src_feats_reshaped, top_10_kmeans[value, :]) for value in range(0, int(queries_per_round/no_batches))]
                    top_k_ent_indices = np.asarray(nearest_indices )


                elif args.query_type == 3:
                    # infonn method

                    # distribution parameters
                    num_samples = 100
                    clustering_choice = 0

                    # compute the embeddings
                    all_indices = np.arange(len(x_src_feats_reshaped))
                    ind_l = pool[btch,:].astype(int)
                    ind_u = list(set(all_indices) - set(ind_l))

                    embeddings_u = x_src_feats_reshaped[ind_u, :]
                    embeddings_l = x_src_feats_reshaped[ind_l, :]
                    y_src = data['y'].reshape(-1,).to(device)
                    labels = y_src[ind_l]

                    # compute the possible queries
                    pseudo_labels, candidate_queries, dist_std, mu = form_queries(embeddings_u, embeddings_l, labels)
                #     mu = mu*(0.99**iter_num)
                #     print('mu: {}'.format(mu))

                    # select the optimal query
                    infogain_u = []

                    for i in range(candidate_queries.shape[0]):
                        temp = mutual_information(device, candidate_queries[i], num_samples, dist_std, mu)
                        infogain_u.append(temp)

                    infogain_u = torch.stack(infogain_u)
                #     next_samples = torch.topk(infogain_u, n_samples, largest=True)[1]

                    num_clusters = 5

                    if clustering_choice:
                        # use knn
                        cluster_ids = pseudo_labels
                        samples_u = torch.cat((cluster_ids.float().reshape(-1,1), infogain_u.reshape(-1,1)), dim=1)

                    else:
                #         use kmeans
                        cluster_ids, cluster_centers = kmeans_fit(X=embeddings_u, num_clusters=num_clusters, distance='euclidean', device=device)
                        samples_u = torch.cat((cluster_ids.float().reshape(-1,1), infogain_u.reshape(-1,1)), dim=1)

                    next_samples = []
                    num_unlabeled = cluster_ids.shape[0]
                    for k in range(num_clusters):
                        mask = samples_u[:,0] == k
                        true_ind = torch.nonzero(mask, as_tuple=True)
                        cluster_size = true_ind[0].size()[0]
                        num_per_cluster = math.ceil((cluster_size * n_samples) / num_unlabeled)
                        _, pseudo_ind = torch.topk(samples_u[true_ind][:,1], num_per_cluster, largest=True)
                        topk_true_ind = true_ind[0][pseudo_ind]
                        next_samples.append(topk_true_ind)
                    next_samples = torch.cat(next_samples)
                    next_samples = next_samples.cpu().detach().numpy()
                    top_k_ent_indices = np.ndarray.flatten(np.squeeze(next_samples))

                elif args.query_type == 4:
                    # coreset
                    # compute the embeddings
                    all_indices = np.arange(len(x_src_feats_reshaped))
                    ind_l = pool[btch, :].astype(int)
                    ind_u = list(set(all_indices) - set(ind_l))

                    embeddings_u = x_src_feats_reshaped[ind_u, :]
                    embeddings_l = x_src_feats_reshaped[ind_l, :]

                    next_samples = []
                    for _ in range(n_samples):
                        distances = torch.cdist(embeddings_l, embeddings_u)
                        min_dist = torch.min(distances, dim = 0, keepdim = True)[0]
                        ind = torch.argmax(min_dist)
                        next_samples.append(ind)
                        embeddings_l = torch.cat((embeddings_l,embeddings_u[ind].reshape(1,-1)))
                        embeddings_u = torch.cat((embeddings_u[:ind],embeddings_u[ind+1:]))

                    next_samples = torch.stack(next_samples)
                    next_samples = next_samples.cpu().detach().numpy()
                    top_k_ent_indices = np.ndarray.flatten(np.squeeze(next_samples))

                # get new queries for batch
                pool_sample_add[btch, :] = top_k_ent_indices[0:int(queries_per_round/no_batches)]
                btch = btch + 1

            'Add acquired samples to pool'



            visualize = 0
            # plot representations and entropy after training round (only for the first run if multiple runs)
            if visualize == 1 and args.run_no == 0 and (pool.reshape(-1,).shape[0])%10 == 0:
                i = 0
                list_train_loader = list(train_dataloader)
                x = list_train_loader[0]['x'][i]
                x2 = list_train_loader[0]['x'][i+1]
                x3 = list_train_loader[0]['x'][i + 2]
                x4 = list_train_loader[0]['x'][i + 3]
                y = list_train_loader[0]['y'][i]
                y2 = list_train_loader[0]['y'][i+1]
                y3 = list_train_loader[0]['y'][i + 2]
                y4 = list_train_loader[0]['y'][i + 3]
                x = torch.cat((x,x2,x3,x4),axis=0)
                y = torch.cat((y,y2,y3,y4),axis=0)
                if args.query_type == 0:
                    top_10_kmeans = None
                fig = plot_ts_reps(model_feats=model_feats, model_clfr=model_clfr, x=x.numpy(), y=y.numpy(), device=device, clusters =top_10_kmeans,
                                   window=-1, title='str(i)')
                wanb_con.log({f"RepresetSubplots/Val/{args.dict_type[args.query_type]}/ after pool of size  {pool.reshape(-1,).shape[0]}": fig})
                print("here")

            # Add queries to batch and then train again
            pool = np.concatenate((pool, pool_sample_add), axis=1)

        #set the next round to train if it was set no to train when given initial pool
        train_round = 1
    'Returns f1 score, accuracy score and pool of aquired samples'
    return f1_score_list,acc_score_list,pool



# helper functions for info-nn

def get_embedding(model, device, labeled_loader, unlabeled_loader):

    """
    Find the nearest labeled samples from every class to each unlabeled sample

    Arguments:
        model:
        device:
        labeled_loader:
        unlabeled_loader:

    Returns:
        embeddings_u: embeddings corresponding to the unlabeled data
        embeddings_l: embeddings corresponding to the labeled data
        sorted_labels: labels corresponding to the labeled samples sorted according to their labels
    """

    embeddings_l = []
    labels = []
    embeddings_u = []


    model.eval()
    with torch.no_grad():
        for data, target in labeled_loader:
            feat_l, _ = model(data.to(device, dtype=torch.float),1)
            feat_l = feat_l.squeeze(1)
            embeddings_l.append(feat_l)
            labels.append(target.to(device))
        for data, _ in unlabeled_loader:
            feat_u, _ = model(data.to(device, dtype=torch.float),1)
            feat_u = feat_u.squeeze(1)
            embeddings_u.append(feat_u)

    embeddings_l = torch.cat(embeddings_l)
    embeddings_u = torch.cat(embeddings_u)
    labels = torch.cat(labels).float()
    labels=labels.reshape(-1,1)

    return embeddings_u, embeddings_l, labels



def form_queries(embeddings_u, embeddings_l, labels, n_classes=5, num_neighbors=5, topk = 5, norm = 2):

    """
    Find the nearest labeled samples from every class to each unlabeled sample

    Arguments:
        embeddings_u: embeddings corresponding to the unlabeled samples, of shape (n_u, d)
        embeddings_l: embeddings corresponding to the unlabeled samples, of shape (n_l, d)
        norm: The norm to be used to compute distances.

    Returns:
        candidate_queries: matrix of shape (n_u, num_classes) containing distances between unlabeled samples
                           and the nearest neighbors
    """

    distances = torch.cdist(embeddings_l, embeddings_u)
    dist_std = torch.std(distances)
    print('Dist std: {}'.format(dist_std))
#     mu = torch.mean(distances)
    mu = torch.max(distances)
#     mu = 1e-5
    print('mu: {}'.format(mu))
    labels = labels[:,None]
    distances = torch.cat((labels, distances),1)
    _, ind = torch.topk(distances[:,1:], topk, dim = 0, largest=False)
    neighbours = distances[:,0][ind]
    pseudo_labels = Counter(neighbours).most_common(1)[0][0]

    candidate_queries = []

    for k in range(n_classes):
        mask = distances[:,0] == k
        nearest_neighbors = torch.min(distances[torch.nonzero(mask, as_tuple=True)][:,1:], dim = 0, keepdim = True)
        candidate_queries.append(nearest_neighbors[0])

    candidate_queries = torch.cat(candidate_queries).t()

    queries, _ = torch.topk(candidate_queries, num_neighbors, dim = 1, largest=False)
    print(queries.size())

    return pseudo_labels, queries, dist_std, mu


def class_probabilities(distances, mu):
    """
    Compute the class probabilities

    Inputs:
        distances: The precomputed set of pairwise distances between a and each body object,
        mu: Optional regularization parameter, set to 0.5 to ignore
    returns:
        prob: The probability corresponding to each class
    """

    prob = 1 / (distances**2 + mu)
    prob = prob / torch.sum(prob, dim=1, keepdim=True)

    return prob


def mutual_information(device, query, num_samples, dist_std, mu):
    """
    This method corresponds to the mutual information calculation specified in Section 3.1
    Specifically, the method returns the result of inputting the method parameters into formula (9).

    Arguments:
        X: An Nxd embedding of objects
        head: The head of the tuple comparison under consideration
        body: The body of the tuple comparison under consideration
        num_samples: Number of samples to estimate D_s as described in (A3)
        dist_std: Variance parameter as specified in (A3)
        mu: Optional regularization parameter for the probabilistic response model
    returns:
        information: Mutual information as specified in (9) in Section 3.1
    """

    distances = []

    for i in range(query.shape[0]):

        # mean = query[i].item()
        # size = (num_samples,1)

        distances.append(torch.abs(torch.normal(query[i].item(), dist_std.item(), (num_samples,1))))

    distances = torch.squeeze(torch.stack(distances, dim=2), dim=1)
    distances = distances.to(device)

    probability_samples = class_probabilities(distances, mu)
    entropy_samples = -torch.sum(probability_samples * torch.log2(probability_samples), dim=1)
    expected_probabilities = torch.sum(probability_samples, dim=0, keepdim=True) / num_samples
    entropy  = -torch.sum(expected_probabilities * torch.log2(expected_probabilities))
    expected_entropy = torch.sum(entropy_samples) / num_samples
    information = entropy - expected_entropy
    
    return information





