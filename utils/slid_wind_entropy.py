import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from scipy.special import entr
from utils.knn_entropy import entropy
def get_windows_4_ent(dataset, window=600, stride=5):

    'This function creates a numpy array for rob_idx: rob idx for training'
    'Takes in a tensor and returns a numpy array'

    "Input should be Txd. Output N X wind x d"
    windowed_robs = []
    windowed_robs = np.asarray(windowed_robs)




    if window != -1:
        windowed = dataset.unfold(0,window,stride).transpose(1,2).detach().cpu().numpy()
        #windowed_robs = windowed_robs.reshape(-1, dataset.shape[-1], window).transpose(1,2).cpu().numpy()
    else:
        windowed = dataset

    '''
    for i in rob_idx:
        rob_temp = dataset[i, :, :]
        print(i)
        if window == -1:

            x_windowed =  np.expand_dims(rob_temp,axis=0)
        else:
            x_windowed = sliding_window(rob_temp,window,stride)
        windowed_robs = np.concatenate((windowed_robs, x_windowed), axis=0) if len(windowed_robs) else x_windowed
        '''
    return windowed


def get_entp_score(dataset,bins=30):
    kde_score_array = np.zeros((dataset.shape[0],1))
    for i in range(0,dataset.shape[0]):
        scores = entr(np.histogram(dataset[i,:], bins=bins, range=(-60, 60),
                          density=True)[0])
        #kde_score_array[i] = entropy(dataset[i,:,:],k=10)
        #kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(dataset[i,:,:])
        #scores = np.exp(kde.score_samples(dataset[i,:,:]))
        kde_score_array[i] = np.sum(entr(scores))
    return kde_score_array

def get_jumped_indices(y,idx_array):
        '''Input: takes in n array where condition is met to take random samples and returns a list of tuples'
       Output: Tuples for segments (first val start, second val end). Length of array where condition is met'''
        list_segs = []
        t_len_segs = len(idx_array)
        diff_idx = np.diff(idx_array)
        jump_idx = np.where(diff_idx>10)[0]
        end_idx = jump_idx -1

        for i in range(0,len(jump_idx)-1):
            if i == 0:
                tuple = (idx_array[0],idx_array[jump_idx[i]])
            else:
                tuple = (idx_array[jump_idx[i]+1],idx_array[jump_idx[i+1]])
            list_segs.append(tuple)
        return list_segs,t_len_segs
