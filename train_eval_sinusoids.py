import torch
import numpy as np
from torch.utils.data import DataLoader
from loss_ssl_tcn import hierarchical_contrastive_loss
from models_TCN import  TSEncoder, TSEnc_with_Proj
from data_gen import gen_sinusoid,sliding_window,window_sampler,get_list_sinusoids,Robot_data,get_list_gaussians
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import umap
from utils_tcn import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from utils.robo_dataloader_bams import LeggedRobotsDataset_DA, get_robo_windows
from sklearn.manifold import TSNE
from collections import  OrderedDict
from MAML import  MAML, SSL_MAML
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from skimage.restoration import denoise_tv_bregman
from scipy import signal
def fit_lr(features, y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe
import argparse
from sklearn.metrics import confusion_matrix
from utils.bams_train_linear import train_linear_layer
import pandas as pd
import seaborn as sn


def get_two_views(x,max_train_length=None,temporal_unit = 0):
    if max_train_length is not None and x.size(1) > max_train_length:
        window_offset = np.random.randint(x.size(1) - max_train_length + 1)
        x = x[:, window_offset: window_offset + max_train_length]

    ts_l = x.size(1)
    crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
    crop_left = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l
    crop_eleft = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
    x1_view = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
    x2_view = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
    return x1_view, x2_view, crop_l

def train(train_dataloader, optimizer, no_insteps, val_dataloader, criterion, dataset, train_idx,test_idx,
          tboard_path='./runs',
              file_path_save='./saved_models_ssl/robo_tcn'):
    writer = SummaryWriter(tboard_path)
    temp_params = OrderedDict(model.named_parameters())
    model.train()
    # self.model.backbone.requires_grad_(False)

    # with torch.no_grad():
    #    #self.model.projector_head.weight = nn.Parameter(torch.ones_like(64,64))
    #    sparse_k = 32
    #    self.model.projector_head.weight[sparse_k:,:] = 0
    #    self.model.projector_head.bias[sparse_k:] = 0

    # self.model.projector_head.bias.requires_grad_(False)
    # self.model.projector_head.bias.requires_grad_(True)
    # self.model.projector_head.weight.requires_grad_(True)
    b_val_loss = 1e5
    b_rob_score = 0.0
    b_terrain_score = 0.0
    model_awg = torch.optim.swa_utils.AveragedModel(model)

    for ep in range(0, no_insteps):
        loss_array_train = []
        loss_array_val = []
        print("epoch {}".format(ep))
        model.train()
        for x in train_dataloader:
            # x = x[dof].swapaxes(2,1)
            temp_params = OrderedDict(model.named_parameters())
            optimizer.zero_grad()
            x = x.float()
            x1, x2, crop_l = get_two_views(x)
            optimizer.zero_grad()

            out1 = model(x1.cuda(device), temp_params)
            out2 = model(x2.cuda(device), temp_params)
            out1 = out1[:, -crop_l:]
            out2 = out2[:, :crop_l]
            # out2 = out1[:, -crop_l:] +  torch.normal(0,1,size= out1.shape).cuda(self.device)

            loss1 = criterion(
                out1,
                out2,
                temporal_unit=0
            )  # + 0.0*torch.norm(torch.norm(self.model.projector_head.weight, p=1, dim=0), p=1, dim=0)

            no_channels = out1.shape[2]
            length_seq  = out1.shape[1]

            if length_seq  <50:
                continue
            reps1 = torch.clone(out1.swapaxes(1, 2)).detach().cpu().numpy()
            #reps1 = reps1.reshape(-1, length_seq)
            #tv_reg = denoise_tv_bregman(reps1, weight=30.0, eps=0.001, isotropic=True,
            #                           multichannel=True).T
            #tv_reg = torch.from_numpy(tv_reg.reshape(-1,length_seq,no_channels)).cuda(device)

            f1, t1, Zxx = signal.stft(reps1, fs=1, nperseg=50)
            Zxx[:, :,18:, :] = 0
            _, xrec1 = signal.istft(Zxx, fs=1)
            xrec1 = torch.from_numpy(xrec1.swapaxes(1,2)).cuda(device)
            xrec1 = xrec1[:,0:length_seq,:]
            loss2 = criterion(
                out1,
                xrec1,
                temporal_unit=0
            )


            loss =1*loss1 + 1*loss2
            loss.backward()
            optimizer.step()
            model_awg.update_parameters(model)
            loss_array_train.append(loss.item())
        print("train loss {}".format((np.mean(loss_array_train))))
        writer.add_scalar('Loss/train', np.mean(loss_array_train), ep)
        model.eval()
        for x in val_dataloader:
            temp_params = OrderedDict(model.named_parameters())
            optimizer.zero_grad()
            x = x.float()
            x1, x2, crop_l = get_two_views(x)

            out1 = model(x1.cuda(device), temp_params)
            out2 = model(x2.cuda(device), temp_params)
            out1 = out1[:, -crop_l:]
            out2 = out2[:, :crop_l]
            # out2 = out1[:, -crop_l:] +  torch.normal(0,1,size= out1.shape).cuda(self.device)

            loss1 = criterion(
                out1,
                out2,
                temporal_unit=0
            )  # + 0.0*torch.norm(torch.norm(model.projector_head.weight, p=1, dim=0), p=1, dim=0)
            no_channels = out1.shape[2]
            length_seq = out1.shape[1]

            if length_seq  <50:
                continue
            reps1 = torch.clone(out1.swapaxes(1, 2)).detach().cpu().numpy()
            #reps1 = reps1.reshape(-1, length_seq)
            #tv_reg = denoise_tv_bregman(reps1, weight=30.0, eps=0.001, isotropic=True,
            #                            multichannel=True).T
            #tv_reg = torch.from_numpy(tv_reg.reshape(-1, length_seq, no_channels)).cuda(device)
            f1, t1, Zxx = signal.stft(reps1, fs=1, nperseg=50)
            Zxx[:, :, 18:, :] = 0
            _, xrec1 = signal.istft(Zxx, fs=1)
            xrec1 = torch.from_numpy(xrec1.swapaxes(1, 2)).cuda(device)
            xrec1 = xrec1[:, 0:length_seq, :]
            loss2 = criterion(
                out1,
                xrec1,
                temporal_unit=0
            )

            loss = 1*loss1 + 1*loss2
            loss_array_val.append(loss.item())
        writer.add_scalar('Loss/val', np.mean(loss_array_val), ep)
        print("val loss {}".format((np.mean(loss_array_val))))

        if ep % 20 == 0:
            f1_score_list = test(model, device, dataset, train_idx, test_idx)
            print(f1_score_list)
            if f1_score_list[0] >= b_rob_score:
                b_rob_score = f1_score_list[0]
            if f1_score_list[1] >= b_terrain_score:
                b_terrain_score = f1_score_list[1]
        if np.mean(loss_array_val) <= b_val_loss:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, file_path_save)
            b_val_loss = np.mean(loss_array_val)
    print(f"Best Robo Score {b_rob_score}")
    print(f"Best Terrain Score {b_terrain_score}")
    return np.mean(loss_array_train)
def test(model, device, dataset, train_idx, test_idx):
    # get embeddings
    params = OrderedDict(model.named_parameters())

    data_inp = dataset.data[dof].swapaxes(2,1)
    embeddings = model(torch.from_numpy(data_inp).cuda(device),params)
    model.eval()
    # decode from all three embeddings
    def decode(embeddings, target):

        emb_size = embeddings.shape[-1]
        train_data = [embeddings[train_idx].reshape(-1, emb_size).detach(), target[train_idx].reshape(-1)]
        test_data = [embeddings[test_idx].reshape(-1, emb_size).detach(), target[test_idx].reshape(-1)]
        f1_score, cm = train_linear_layer(target.max()+1, train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return f1_score, cm
    f1_score_list =[]
    #for emb_keys in [['recent_past', 'short_term', 'long_term'], ['recent_past'], ['short_term'], ['long_term']]:
    for target_tag in ['robot_type', 'terrain_type', 'terrain_type_finer']:
        target = torch.LongTensor(dataset.data[target_tag])
        if target_tag == 'robot_type':
            target = target.repeat(dataset.data[dof].shape[-1], 1).T
        f1_score, cm = decode(embeddings, target)
        print(f1_score)
        print("here")
        #emb_tag = '_'.join(emb_keys)

        #writer.add_scalar(f'test/f1_', f1_score, epoch)
        class_names = dataset.class_names[target_tag]
        #writer.add_figure(f'{target_tag}',
        #                  sn.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True).get_figure(),
        #                             epoch)
        f1_score_list.append(f1_score)
    return f1_score_list


def custom_test(model, device, dataset, train_idx, test_idx):
    # get embeddings
    params = OrderedDict(model.named_parameters())

    data_inp = dataset.data[dof].swapaxes(2,1)
    embeddings = model(torch.from_numpy(data_inp).cuda(device),params)
    model.eval()
    # decode from all three embeddings
    def decode(embeddings, target):

        emb_size = embeddings.shape[-1]
        train_data = [embeddings[train_idx].reshape(-1, emb_size).detach(), target[train_idx].reshape(-1)]
        test_data = [embeddings[test_idx].reshape(-1, emb_size).detach(), target[test_idx].reshape(-1)]
        f1_score, cm = train_linear_layer(target.max()+1, train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return f1_score, cm
    f1_score_list =[]
    #for emb_keys in [['recent_past', 'short_term', 'long_term'], ['recent_past'], ['short_term'], ['long_term']]:
    for target_tag in ['terrain_type']:
        target = torch.LongTensor(dataset.data[target_tag])
        if target_tag == 'robot_type':
            target = target.repeat(dataset.data[dof].shape[-1], 1).T
        f1_score, cm = decode(embeddings, target)
        print(f1_score)
        print("here")
        #emb_tag = '_'.join(emb_keys)

        #writer.add_scalar(f'test/f1_', f1_score, epoch)
        #class_names = dataset.class_names[target_tag]
        #writer.add_figure(f'{target_tag}',
        #                  sn.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True).get_figure(),
        #                             epoch)
        f1_score_list.append(f1_score)
    return f1_score,cm
def plot_subplot_gifs(transformed,start_index,end_index,label_array,scale,file_path):
    '''this function takes in embeddings for temporal data, and plots a GIF consisting of 2 subplots
    The first subplot consists of a scatter plot colored by temporal instance
    The second subplot consists of a scatter plot colored by class labels
    Each snap shot of the GIF plots these two subplots at every "scale_th" frame

    Inputs: Transformed (reprsentatioin data)
    start_index: starting point for temporal plot
    end_index: end point for temporal plot
    scale: number of consecutive scatter points drawn in each frame
    labels: label array
    file_path : file_path to save'''

    def init():
        print('hello')
        ax1.set_xlim(-15,18)
        ax1.set_ylim(-15, 18)
        ax2.set_xlim(-15, 18)
        ax2.set_ylim(-15, 18)

    def update(i):
        t= np.arange(start_index,end_index)
        pos_neg = ax1.scatter(transformed[start_index:i, 0], transformed[start_index:i, 1],c=t[:i-start_index], \
        vmin=start_index,vmax=end_index)

        #ax1.set_title("Uptill time {}".format(i))
        if i == start_index:
            plt.colorbar(pos_neg,ax = ax1)
        plt.title("Uptill time {}".format(i))
        plt.tight_layout()
        #plt.title("Umap : {}".format(i))
        #if i[-1] == np_points - 1:
        #    plt.colorbar()

        uniq_labels_list = list(np.unique(label_array))
        list_color =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',\
 '#7f7f7f', '#bcbd22', '#17becf','r','c','g','k']
        lbl_list = []
        for id_lbl,lbl in enumerate(uniq_labels_list):

            label_idx = np.where(label_array == int(lbl))[0]
            label_idx = label_idx[(label_idx >= start_index) & (label_idx <= i) &(label_idx >= max(0,i-scale))]
            ax2.scatter(transformed[label_idx, 0], transformed[label_idx, 1],label=str(lbl)  ,color=list_color[id_lbl])

            lbl_list.append(str(lbl))
        plt.title("Uptill time {}".format(i))
        '''
        one_labels = np.where(label_array == 1)[0]
        one_labels = one_labels[(one_labels >= start_index) & (one_labels <= i)]

        two_labels = np.where(label_array == 2)[0]
        two_labels = two_labels[(two_labels >= start_index) & (two_labels <= i)]
        ax2.scatter(transformed[zer_labels, 0], transformed[zer_labels, 1], color='#1f77b4',
                    label='0')

        ax2.scatter(transformed[one_labels, 0], transformed[one_labels, 1], color='#ff7f0e',
                    label='1')
        ax2.scatter(transformed[two_labels, 0], transformed[two_labels, 1], color='#2ca02c',
                    label='2')
        '''
        if i == start_index:
            ax2.legend(lbl_list)
        plt.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,10))
    t = np.arange(start_index,end_index,scale)
    plt.xlim(-15, 18)
    plt.ylim(-15, 18)
    ani1 = FuncAnimation(fig, update, frames=list(t), init_func=init)
    writer = PillowWriter(fps=3)
    ani1.save(file_path+'.gif', writer=writer)
    fig.savefig(file_path+'.png')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',type=str,default='robots',help='Dataset')
parser.add_argument('--model_pth',type=str,default='saved_models_ssl/ssl_save_robots_no_fixed',help='file path to saved model')
parser.add_argument('--gpu',type=int,default=4,help='gpu for device')
parser.add_argument('--semi_sup',type=int,default=0,help='Semi sup represetation')
parser.add_argument('--robot_type', type=str,default='b', help='If dataset is robot, what robot type')
parser.add_argument('--finetuning', type=int,default=0, help='Fine tune to task')
parser.add_argument('--meta_learn_reps', type=int,default=0, help='Meta learn represenentations')
parser.add_argument('--model_with_proj', type=int,default=0, help='modle encoder linear proj')
parser.add_argument('--load', type=int,default=1, help='load model')
parser.add_argument('--fine_tune_steps', type=int,default=300, help='fine_tune_steps')
parser.add_argument('--augment_hetro', type=int,default=0, help='augment hetro data')
parser.add_argument('--tboard_path', type=str,default='./runs/hetro/', help='augment hetro data')
parser.add_argument('--batch_size', type=int,default=16, help='augment hetro data')
parser.add_argument('--train', type=int,default=0, help='train or not')
parser.add_argument('--dof_type', type=str,default='pos', help='dof vel or pos')
args = parser.parse_args()
dof = args.dof_type
device =  torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

'select robot indice and dof value to visualize'
rob_no = 1
dof_idx = np.arange(0,10)


if str.lower(args.dataset) == 'sinusoids':
    saw_tooth = 0
    meta_model = 1
    if meta_model:
        file_path_save = 'saved_models_ssl/ssl_save'
        file_path_str  = 'figures/metamodel'
    else:
        file_path_save = 'saved_models_ssl/ssl_save'
        file_path_str = 'figures/TSNE_10_base_task'
    loss_crietrion = hierarchical_contrastive_loss

    #file_path_save_backbone = 'saved_models_ssl/ssl_save_only10'
    #file_path_save_backbone = 'saved_models_ssl/ssl_save_sinusoids'
    #file_path_save_backbone = 'saved_models_ssl/ssl_save_sinusoids_fixed_head'
    file_path_save = 'saved_models_ssl/ssl_save_sinusoids_l1_proj'
    enc_out_dims = 64
    hidden_dims = 64
    proj_dims = 64
    input_dims = 1

    hidden_dims = 64
    depth = 10
    batch_size = args.batch_size
    if args.model_with_proj:
        model = TSEnc_with_Proj(input_dims=input_dims, enc_out_dims=enc_out_dims, project_dims= proj_dims,hidden_dims=hidden_dims, depth=depth)
    else:
        model = TSEncoder(input_dims=input_dims, output_dims=enc_out_dims,
                         hidden_dims=hidden_dims, depth=depth)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if args.load == 1:
        checkpoint = torch.load(file_path_save)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.cuda(device)
    base_freq = 20
    Fs = 500
    change = 1
    #x_series,y = gen_sinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=10)
    freq_list =  [11,12,13,14]
    down_stream_freqs = freq_list # [12, 10, 13, 20]
    freq_list_test =  freq_list #[12, 10, 13, 20]
    mu_list =[0,2,3,6,-3,1,2.5,1,-10,0,9]
    var_list = [1,0,1,2,1,1,1,3,10,3,2]
    x_series,y =  get_list_sinusoids(Fs,freq_list ,saw_tooth=saw_tooth,noise=0.3)#gen_lsinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=10)
    mean = np.nanmean(x_series)
    std = np.nanstd(x_series)
    x_series = (x_series - mean) / std
    x_series = x_series.reshape(-1,1)
    x_series2, y2 = get_list_gaussians(Fs, mu_list,var_list)
    mean = np.nanmean(x_series2)
    std = np.nanstd(x_series2)
    #x_series2 = (x_series2 - mean) / std
    x_series2 = x_series2.reshape(-1, 1)

    x_series_val, y_val = get_list_sinusoids(Fs, freq_list, saw_tooth=saw_tooth,
                                     noise=0.1)  # gen_lsinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=10)
    mean = np.nanmean(x_series_val)
    std = np.nanstd(x_series_val)
    x_series_val = (x_series_val - mean) / std
    x_series_val = x_series_val.reshape(-1, 1)
    x_series2_val, y2_val = get_list_gaussians(Fs, mu_list, var_list)
    mean = np.nanmean(x_series2_val)
    std = np.nanstd(x_series2_val)
    # x_series2 = (x_series2 - mean) / std
    x_series2_val = x_series2_val.reshape(-1, 1)
    y2 = 1+y2 + np.max(y)
    if args.augment_hetro:
        x_series = np.concatenate((x_series, x_series2), axis=0)
        x_series_val = np.concatenate((x_series_val, x_series2_val), axis=0)
        #y = np.concatenate((y,y2),axis=0)
    window = 200
    stride = 10
    x_windowed = sliding_window(x_series, window, stride)
    fine_tune_dataset = window_sampler(x_windowed)
    fine_tune_dldr = DataLoader(fine_tune_dataset,batch_size=batch_size,shuffle=True)

    x_windowed_val = sliding_window(x_series_val, window, stride)
    fine_tune_dataset_val = window_sampler(x_windowed_val)
    fine_tune_dldr_val = DataLoader(fine_tune_dataset_val, batch_size=batch_size, shuffle=True)

    lr_out = 0.0001
    lr_in = 0.001
    no_insteps = args.fine_tune_steps
    optim = torch.optim.Adam(model.parameters(),lr_out )
    #maml_obj = SSL_MAML(model=model,optimizer=optim,criterion=hierarchical_contrastive_loss,lr_in=lr_in,no_insteps=no_insteps,device=device)
    if args.finetuning:
        loss = train(train_dataloader=fine_tune_dldr, optimizer=optim, no_insteps=no_insteps,
                     val_dataloader=fine_tune_dldr_val, tboard_path=args.tboard_path)
    params = OrderedDict(model.named_parameters())


    'done tuning'
    x_series = x_series.reshape(1, -1, 1)

    x_series_dstream, y_dstream = get_list_sinusoids(Fs, down_stream_freqs,
                                             saw_tooth=saw_tooth,noise =0.3)  # gen_lsinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=10)
    #x_series_dstream  = x_series
    #y_dstream = y
    mean = np.nanmean(x_series_dstream)
    std = np.nanstd(x_series_dstream)
    x_series_dstream = (x_series_dstream - mean) / std
    x_series_dstream = x_series_dstream.reshape(1, -1, 1)
    if args.meta_learn_reps and args.model_with_proj:
        train_rep_dstream = model.get_backbone(torch.from_numpy(x_series_dstream).float().cuda(1), params)[0, :, :].detach().cpu()
    else:
        train_rep_dstream = model(torch.from_numpy(x_series_dstream).float().cuda(1), params)[0, :, :].detach().cpu()

    #train_repr = model(torch.from_numpy(x_series).float().cuda(1),params)[0,:,:]
    #all_reps = model.all_layer_TCN_out(torch.from_numpy(x_series).float().cuda(1),params)
    #feats = model.get_backbone(torch.from_numpy(x_series).float().cuda(1), params)
    train_rep_dstream = train_rep_dstream.detach().cpu().numpy()


    clf = fit_lr(train_rep_dstream, y_dstream)

    acc = clf.score(train_rep_dstream, y_dstream)
    y_pred = clf.predict(train_rep_dstream)
    #c_mtrx = confusion_matrix(y, y_pred)

    x_series_tst, y_tst = get_list_sinusoids(Fs, freq_list_test,
                                     saw_tooth=saw_tooth)  # gen_lsinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=10)
    mean = np.nanmean(x_series_tst)
    std = np.nanstd(x_series_tst)
    x_series_tst = (x_series_tst - mean) / std
    x_series_tst = x_series_tst.reshape(1, -1, 1)
    test_repr = model(torch.from_numpy(x_series_tst).float().cuda(1), params)[0, :, :].detach().cpu()
    y_pred_tst = clf.predict(test_repr)
    c_mtrx_tst = confusion_matrix(y_tst , y_pred_tst )
    acc_tst = clf.score(test_repr, y_tst)
    reducer = umap.UMAP()
    # reducer = TSNE(n_components=2, learning_rate= 6.2,
    #            init='random', perplexity=3)
    # reducer = PCA(n_components=2, svd_solver='full')
    transformed = reducer.fit_transform(train_rep_dstream)
    if saw_tooth == 1:
        file_path = file_path_str+'_sawtooth_{}'.format(str(freq_list))
    else:
        file_path = file_path_str+'_sinusoid_{}'.format(str(freq_list))
    #file_path = 'figures/10base_model_only_finetuned!{}'.format(base_freq)
    #file_path = 'figures/metamodel_finetuned{}'.format(str(freq_list))
    #file_path = 'figures/10base_finetuned{}'.format(str(freq_list))
    start_index = 0
    np_points = min(4000,len(transformed))
    np_points = len(transformed)
    plot_subplot_gifs(transformed, start_index, np_points, y, scale=500, file_path=file_path)
    print(freq_list)
elif str.lower(args.dataset) == 'robots':

    #file_path_save = f'./saved_models_ssl/robo_tcn/robo_TS2Vec'
    #file_path_save = f'./saved_models_ssl/robo_tcn/robo_TS2Vec_bbone'
    file_path_save = f'./saved_models_ssl/robo_tcn/robo_TS2Vec_bbone__WD'
    input_dims = 12
    output_dims = 6
    hidden_dims = 66
    depth = 5
    enc_out_dims = 66
    batch_size = args.batch_size
    proj_dims = 66

    #model = TSEnc_with_Proj(input_dims=input_dims, enc_out_dims=enc_out_dims, project_dims= proj_dims,hidden_dims=hidden_dims, depth=depth)
    model = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                      hidden_dims=hidden_dims, depth=depth)
    if args.load:
        checkpoint = torch.load(file_path_save)
        model.load_state_dict(checkpoint['model_state_dict'])
    lr_out = 0.001
    lr_in = 0.01
    no_insteps = 10
    optim = torch.optim.Adam(model.parameters(), lr_out)
    #maml_obj = SSL_MAML(model=model, optimizer=optim, criterion=hierarchical_contrastive_loss, lr_in=lr_in,
    #                    no_insteps=no_insteps, device=device)
    #checkpoint = torch.load(args.model_pth)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    params = OrderedDict(model.named_parameters())


    if str.lower(args.robot_type) == 'b':
        data_path = './updated_robot_data/saved_anymal_b_revised.npz'
        label_path = './updated_robot_data/saved_anymal_labels_robot_b.npz'
    elif str.lower(args.robot_type) == 'c':
        data_path = './updated_robot_data/saved_anymal_c_revised.npz'
        label_path = './updated_robot_data/saved_anymal_labels_robot_c.npz'
   # rbt_data = Robot_data(file_path=data_path, file_path_labels=label_path, task_types='both')
    data_path = './updated_robot_data/legged_robots_v1.npy'
    dataset = LeggedRobotsDataset_DA(data_path,src_list=list(np.arange(0,76)),trg_list=list(np.arange(0,76)),dof=dof)

    train_idx, test_idx = train_test_split(np.arange(38), test_size=0.2, random_state=42)

    #robo_win_train = get_robo_windows(dataset.data[dof],train_idx)
    #robo_win_test = get_robo_windows(dataset.data[dof], test_idx)
    robo_win_train = dataset.data[dof][train_idx,:,:].swapaxes(1,2)
    robo_win_test = dataset.data[dof][test_idx,:,:].swapaxes(1,2)
    train_loader = DataLoader(robo_win_train, batch_size=batch_size, drop_last=False)
    val_loader = DataLoader(robo_win_test, batch_size=batch_size, drop_last=False)
    if args.train:
        loss = train(train_dataloader=train_loader, no_insteps=args.fine_tune_steps, optimizer=optim,
                     criterion=hierarchical_contrastive_loss,val_dataloader=val_loader,
                                 dataset=dataset,train_idx= train_idx,test_idx=test_idx,tboard_path=args.tboard_path,file_path_save=file_path_save)
    #repr_dof_hldout = model(torch.from_numpy(x_series_dof_pos_hldout ).float().to(device), params)[0, :, :]
    test_data = dataset.data[dof][test_idx,:,:]
    test_data = test_data.swapaxes(1,2)
    for target_tag in ['robot_type', 'terrain_type']:
        print(target_tag)
    'Checking generalization when linear readouts trained on one type of robot and test on another'

    'Trained on robot B, Tested on C'

    train_idx = np.arange(0, 20)
    test_idx = np.arange(38, 58)

    f1_trn_B_test_B, cm = custom_test(model, device, dataset, train_idx, test_idx)

    train_idx = np.concatenate((np.arange(0,20),np.arange(38,58)),axis=0)
    test_idx = np.concatenate((np.arange(20,38),np.arange(58,76)),axis=0)
    f1_trn_B_test_C,cm = custom_test(model, device, dataset, train_idx, test_idx)
    #print(f"Terrain score train on B, test on C {f1_trn_B_test_C[0]}")

    'Trained on robot C, Tested on B'
    #train_idx = np.arange(38, 76)
    #test_idx = np.arange(0, 38)
    train_idx = np.arange(38, 58)
    test_idx = np.arange(58, 76)
    f1_trn_C_test_B,cm = custom_test(model, device, dataset, train_idx, test_idx)
    #print(f"Terrain score train on C, test on B {f1_trn_C_test_B[0]}")

    'Trained on robot B, Tested on held out B'
    train_idx = np.arange(0, 20)
    test_idx = np.arange(20, 38)

    f1_trn_B_test_B,cm = custom_test(model, device, dataset, train_idx, test_idx)
    #print(f"Terrain score train on B, test on held out B {f1_trn_B_test_B[0]}")

    'Trained on robot C, Tested on held out C'
    train_idx = np.arange(40, 60)
    #test_idx = np.arange(58, 76)
    test_idx = np.arange(60, 76)
    f1_trn_C_test_C,cm = custom_test(model, device, dataset, train_idx, test_idx)
    print(f"Terrain score train on C, test on held out C {f1_trn_C_test_C[0]}")
    '''
    clf = fit_lr(repr_dof, y_label)
    acc = clf.score(repr_dof, y_label)
    acc = clf.score(repr_dof_hldout, y_label_hldout )

    print("Accuracy on Robot for Dof POS {0}: {1}".format(rob_no+2,acc))


    'getting dof vel reps'
    reducer = umap.UMAP()
    x_series_dof_vel = x_series_dof_vel.reshape(1, -1, 1)
    x_seroes_dof_vel_hldout = x_series_dof_vel.reshape(1,-1,1)
    repr_vel = model(torch.from_numpy(x_series_dof_vel).float().to(device), params)[0, :, :]
    repr_vel_hldout = model(torch.from_numpy(x_seroes_dof_vel_hldout).float().to(device), params)[0, :, :]
    repr_vel = repr_vel.detach().cpu().numpy()
    repr_vel_hldout = repr_vel_hldout.detach().cpu().numpy()


    clf = fit_lr(repr_vel , y_label)
    acc = clf.score(repr_vel_hldout, y_label_hldout )

    print("Accuracy on Robot for Dof Vel {0}: {1}".format(rob_no + 2, acc))

    ###########################################

    umap_repr_dof = reducer.fit_transform(repr_dof)
    umap_repr_vel = reducer.fit_transform(repr_vel)

    file_path_dof = 'figures/robots/{}__POS_dof_no_{}_robot_rob_no_{}'.format(str.upper(args.robot_type),dof_idx,rob_no)
    plot_subplot_gifs(umap_repr_dof, start_index=0, end_index=umap_repr_dof.shape[0], label_array=y_label, \
                      scale=100, file_path=file_path_dof)

    file_path_vel = 'figures/robots/{}__VEL_dof_no_{}_robot_rob_no_{}'.format(str.upper(args.robot_type), dof_idx, rob_no)
    plot_subplot_gifs(umap_repr_vel, start_index=0, end_index=umap_repr_vel.shape[0], label_array=y_label, \
                      scale=100, file_path=file_path_vel)
                      '''