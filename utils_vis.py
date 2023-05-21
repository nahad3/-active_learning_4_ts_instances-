import matplotlib.pyplot as plt
import numpy as np
#import umap
import torch
from collections import OrderedDict
from torch import nn
from numpy import linalg as LA

def find_nearest(array, value):
    'takes argmin l1 norm for array - value (closest l1 norm index to kmeans centeroid)'
    array = np.asarray(array)
    idx = (LA.norm((array - value),axis=1)).argmin()
    return idx


def get_invariant_feats(model_feats,model_da, device, test_loader,use_ssl=1,only_ssl=1,no_da=0):
    'no_da: no domain adaptation'
    model_feats.eval()
    model_da.eval()
    test_loader
    alpha = 1
    pred_array = []
    y_lab_array = []
    params = OrderedDict(model_feats.named_parameters())
    #params = params.to(device)
    pred_array = np.asarray(pred_array)
    y_lab_array = np.asarray(y_lab_array)
    with torch.no_grad():
        for data in test_loader:
            x_src = data['x_src'].to(device).to(device)
            y_src = data['y_src'].to(device).to(device)
            if use_ssl:



                x_src_feats = model_feats(x_src, params)
                #x_src_feats = x_src[:,:,:]
                #reps = model(x,params).detach().cpu().numpy().squeeze(0)




                #x_src_feats = model_feats(z, params)
                #x_src_feats = x_src_feats.reshape(x_src.shape[0],-1,x_src_feats.shape[-1])
            else:
                x_src_feats = x_src
            if no_da:
                params_da =  OrderedDict(model_feats.named_parameters())
                invrt_feats = model_da(x_src_feats,params_da )
            elif only_ssl:
                invrt_feats = x_src_feats
            else:
                invrt_feats = model_da.get_invariant_feats(x_src_feats)


    return invrt_feats,y_src
def visualize_reps(model_feats,model_da,device,test_loader,rob_c_idx,rob_b_idx,use_ssl=1,only_ssl=1,src_type ='B',trgt_type='C',no_da=0):
    reducer = umap.UMAP()
    #reducer = TSNE(n_components=2)
    if only_ssl:
        assert use_ssl == 1, "use ssl should be 1 if only ssl is 1"
    reps,y_src = get_invariant_feats(model_feats = model_feats, model_da=model_da,device=device,test_loader=test_loader,
                                     use_ssl=use_ssl,only_ssl=only_ssl)
    #reps = reps[:,:,0:32]
    if use_ssl == 1:
        ssl_sv_str = 'with_ssl'
    else:
        ssl_sv_str = 'without_ssl'
    if only_ssl == 1:
        only_ssl_str = 'with_only_ssl'
    else:
        only_ssl_str = 'without_ssl'

    y_src = y_src.cpu().numpy()
    rob_lbl = np.zeros(y_src.shape)
    rob_lbl[rob_b_idx,:,:] = 1
    y_src2= np.copy(y_src)
    y_src2[rob_b_idx,:,:] = y_src2[rob_b_idx,:,:] + 10
    reps = reps.cpu().numpy().reshape(-1,reps.shape[-1])
    rob_lbl = rob_lbl.reshape(-1,rob_lbl.shape[-1])
    y_src2 = y_src2.reshape(-1,y_src2.shape[-1])
    y_src = y_src.reshape(-1, y_src.shape[-1])
    transformed = reducer.fit_transform(reps)
    save_path_rob = f'figures/DA/rob_lab_{ssl_sv_str}_{only_ssl_str}_src_{src_type}_trgt_{trgt_type}.pdf'
    save_path_terr = f'figures/DA/rob_ter_{ssl_sv_str}_{only_ssl_str}_src_{src_type}_trgt_{trgt_type}.pdf'
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    axs1[0].scatter(transformed[:, 0], transformed[:, 1], c=rob_lbl,alpha=0.1)

    axs1[1].scatter(transformed[(rob_lbl == 0).reshape(-1,), 0], transformed[(rob_lbl == 0).reshape(-1,), 1], c='tab:blue', alpha=0.1,label='Robot c')
    axs1[1].legend()
    axs1[2].scatter(transformed[(rob_lbl == 1).reshape(-1,), 0], transformed[(rob_lbl == 1).reshape(-1,), 1], c='tab:orange', alpha=0.1,label='Robot b')
    axs1[2].legend()
    axs2[0].scatter(transformed[:, 0], transformed[:, 1], c=y_src2,
                    label=['b0', 'b1', 'b2', 'b3', 'b4', 'C0', 'C1', 'C2', 'C3', 'C4'], alpha=0.2)
    clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']
    label_list = ['Stair Down', 'flat', 'Stair up', 'Slope down', 'Slope Up']
    for j in range(0, 5):
        j_idx = np.where(((rob_lbl == 0) & (y_src == j)))[0]
        axs2[1].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j], label=label_list[j],alpha=0.2)
    for j in range(0, 5):
        j_idx = np.where(((rob_lbl == 1)  &(y_src == j)))[0]
        axs2[2].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j], label=label_list[j],alpha=0.2)
    axs2[1].set_title("Robot C")
    axs2[2].set_title("Robot B")
    axs2[0].set_title("Combined Robots")
    axs2[2].legend()
    '''
    for i in range(0,reps.shape[0]):
        temp_reps = reps[i, :, :]
        if (i not in rob_c_idx) and ( i not in rob_b_idx) :
            continue
        else:
            transformed = reducer.fit_transform(temp_reps)
        if i in list(rob_b_idx):
            lbl = 'rob_b'
            clr ='b'
            clr_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:brown']
            label_list = ['B1','B2','B3','B4','B5']
        elif i in list(rob_c_idx):
            lbl = 'rob_c'
            clr = 'r'
            clr_list = ['tab:pink' , 'tab:gray', 'tab:olive','tab:cyan','k']
            label_list = ['C1', 'C2', 'C3', 'C4','C5']
        for j in range(0,5):
            j_idx = np.where(y_src[i,:].reshape(-1,) == j)
            axs[0].scatter(transformed[j_idx,0],transformed[j_idx,1],c= clr_list[j],label = label_list[j])
        for k in range(10,15):
            j_idx = np.where(y_src[i, :].reshape(-1, ) == j)
            axs[0].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j], label=label_list[j])
        axs[1].scatter(transformed[:,0],transformed[:,1],c = clr,label = lbl)
        '''
    fig1.suptitle(f'Robot Type - Src {src_type} Trgt {trgt_type}')
    fig2.suptitle(f'Terrai Type - Src {src_type} Trgt {trgt_type}')
    fig1.savefig(save_path_rob)
    print(save_path_terr)
    return fig2
    #fig2.savefig(save_path_terr)

def visualize_reps_with_labels(reps,y_unlab, y_src,y_prop=None):
    #reps = reps.cpu().numpy().reshape(-1, reps.shape[-1])
    reducer = umap.UMAP()
    y_src = y_src.reshape(-1, )
    transformed = reducer.fit_transform(reps)
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5))



    clr_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown']
    label_list = ['Stair Down', 'flat', 'Stair up', 'Slope down', 'Slope Up']
    for j in range(0, 5):
        j_idx = np.where( y_src == j)[0]
        axs1[1].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j], label=label_list[j], alpha=0.2)
    for j in range(0, 5):
        j_idx = np.where((y_prop == j))[0]
        axs1[2].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j], label=label_list[j], alpha=0.2)


    clr_list = ['k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown',
            ]
    label_list = ['unlabelled','Stair Down', 'flat', 'Stair up', 'Slope down', 'Slope Up',]

    for j in range(-1, 5):
        j_idx = np.where((y_unlab == j))[0]
        axs1[0].scatter(transformed[j_idx, 0], transformed[j_idx, 1], c=clr_list[j+1], label=label_list[j+1], alpha=0.2)
    plt.savefig(f'figures/DA/rob_lab_propagated_vel_b_01')
def plot_ts_reps(model_feats,model_clfr, device, x,y,title='figure',clusters=None,window=50):
    #fig1, axs1 = plt.subplots(4, 1, figsize=(15, 5))
    params = OrderedDict(model_feats.named_parameters())
    model_feats.to(device)
    x = torch.from_numpy(x)[:,:].to(device).unsqueeze(0)

    if window == -1:
        #src_lbl_clfr = model_clfr(x_src_feats)
        reps = model_feats(x,params) #.detach().cpu().numpy().squeeze(0)
        src_lbl_clfr = model_clfr(reps)
        reps = reps.detach().cpu().numpy().squeeze(0)
        src_lbl_clfr_reshaped = src_lbl_clfr.reshape(-1, src_lbl_clfr.shape[-1])
        probs = torch.nn.functional.softmax(src_lbl_clfr_reshaped, dim=1).detach()
        entropy = torch.distributions.Categorical(probs=probs).entropy().cpu().numpy()
        if clusters is not None:
            nearest_indices = [find_nearest(reps.reshape(-1,reps.shape[-1]), clusters[value, :]) for value in
                               range(0, clusters.shape[0])]

        preds = torch.argmax(probs,dim=1).cpu().numpy()
        #reps = x.cpu().numpy().squeeze(0)
    else:
        z = x.squeeze(0).unfold(0, window, window)
        z = z.swapaxes(1, 2)
        reps = model(z, params).detach().cpu()
        reps = reps.reshape(-1,reps.shape[-1])


    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(x[0,:,:].detach().cpu().numpy())
    ax[0].title.set_text("Input data")
    ax[1].plot(reps)
    ax[1].title.set_text("Learned Reps")
    ax[2].plot(entropy)
    if clusters is not None:
        ax[2].vlines(nearest_indices,-0.2,1.5,colors='r')
    ax[2].title.set_text("Entropy (soft max)")
    ax[3].plot(preds)
    ax[3].title.set_text("Predicted class")

    ax[4].plot(y)
    ax[4].title.set_text("True class")
    #plt.show()

    plt.tight_layout()
    fig_path = f'figures/DA/rob_lab_1{title}_{str(window)}.pdf'
    #fig.suptitle(title)
    #plt.show()

    return fig
    #fig.savefig(fig_path)
