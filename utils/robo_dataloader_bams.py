import numpy as np
import os
from torch.utils.data import Dataset
from  itertools import  combinations as comb
import torch
from sklearn.model_selection import train_test_split
from random import sample

def sliding_window(a, win_size,stride_step):
    '''Slding window view of a 2D array a using numpy stride tricks.
        For a given input array `a` and the output array `b`, we will have
        `b[i] = a[i:i+w]`

        Args:
            a: numpy array of shape (N,M)
        Returns:
            numpy array of shape (K,w,M) where K=N-w+1
        '''


    shape = (a.shape[0] - win_size + 1, win_size) + a.shape[-1:]
    strides = (a.strides[0],) + a.strides
    unstrided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    stride_samples = np.linspace(0, unstrided.shape[0], num=int(unstrided.shape[0]/stride_step)).astype(int)
    return unstrided[stride_samples[:-1],:,:]

class LeggedRobotsDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data, _ = self.load()

        self.normalize()

    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        #class_names = data.pop('class_names')

        for key in ['pos', 'vel']:
            data[key] = data[key].astype(np.float32)

        return data, _ # class_names

    def normalize(self):
        pos_mean, pos_std = self.data['pos'].mean(axis=(0,2), keepdims=True), self.data['pos'].std(axis=(0,2), keepdims=True)
        self.data['pos'] = (self.data['pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['vel'].mean(axis=(0, 2), keepdims=True), self.data['vel'].std(axis=(0, 2), keepdims=True)
        self.data['vel'] = (self.data['vel'] - vel_mean) / vel_std

    def __getitem__(self, item):
        data = {key: x[item] for key, x in self.data.items()}
        return data

    def __len__(self):
        return self.data['pos'].shape[0]

class LeggedRobotsDataset_new(Dataset):
    def __init__(self, path,alpha=1):
        self.path = path
        self.alpha = alpha
        self.data = self.load()

        self.normalize()

    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        #class_names = data.pop('class_names')

        #for key in ['dof_pos', 'dof_vel']:
        #    data[key] = data[key].astype(np.float32)

        for key in ['pos', 'dof_vel']:
            data[key] = data[key].astype(np.float32)

        return data

    def normalize(self):
        pos_mean, pos_std = self.data['pos'].mean(axis=(0,2), keepdims=True), self.data['pos'].std(axis=(0,2), keepdims=True)
        self.data['pos'] = (self.data['pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['vel'].mean(axis=(0, 2), keepdims=True), self.data['vel'].std(axis=(0, 2), keepdims=True)
        self.data['vel'] = (self.data['vel'] - vel_mean) / vel_std
        if self.alpha != 1:
            self.data['input'] = np.concatenate([self.data['pos'][:,:,0:4500], self.data['vel'][:,:,0:4500]], axis=1)
            #self.data['input'] = np.concatenate([self.data['pos'][:, :, :], self.data['vel'][:, :, :]], axis=1)
        else:
            self.data['input'] = np.concatenate([self.data['pos'][:, :, :], self.data['vel'][:, :, :]], axis=1)
    def __getitem__(self, item):
        for key,x in self.data.items():
            if key == 'names':
                continue
            else:
                data = {key: x[item]}
        return data

    def __len__(self):
        return self.data['dof_pos'].shape[0]

def get_robo_windows(dataset,rob_idx,window=600,stride=5):

        'This function creates a numpy array for rob_idx: rob idx for training'
        windowed_robs = []
        windowed_robs= np.asarray(windowed_robs)
        dataset = dataset.swapaxes(1,2)
        for i in rob_idx:
            rob_temp = dataset[i,:,:]
            x_windowed = sliding_window(rob_temp, window, stride)
            windowed_robs = np.concatenate((windowed_robs,x_windowed),axis=0) if len(windowed_robs) else x_windowed
        return windowed_robs


import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal


class LeggedRobotsDataset_DA(Dataset):
    def __init__(self, path,src_list,trg_list,window=50,stride=10,dof ='vel',replace_labels_file=-1):
        self.path = path
        self.data = self.load()
        self.dof = dof
        self.normalize()
        self.idx_b,self.idx_c = self.get_src_trgt_indices()

        
        self.x_src,self.y_src,self.x_trgt, self.y_trgt = self.prepare_src_trgt_data(src_idx=src_list,trgt_idx=trg_list,window=window,stride=stride,dof = dof)
        self.no_classes = len(np.unique(self.y_src))
    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        #class_names = data.pop('class_names')

        for key in ['pos', 'vel']:
            data[key] = data[key].astype(np.float32)

        return data

    def normalize(self):
        pos_mean, pos_std = self.data['pos'].mean(axis=(0,2), keepdims=True), self.data['pos'].std(axis=(0,2), keepdims=True)
        self.data['pos'] = (self.data['pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['vel'].mean(axis=(0, 2), keepdims=True), self.data['vel'].std(axis=(0, 2), keepdims=True)
        self.data['vel'] = (self.data['vel'] - vel_mean) / vel_std
    '''
    def __getitem__(self, item):
        data = {key: x[item] for key, x in self.data.items()}
        return data

    def __len__(self):
        return self.data['pos'].shape[0]
    '''

    def __len__(self):
        len = min(self.x_src.shape[0],self.x_trgt.shape[0])

        return len

    def __getitem__(self, item):
        data = {}
        data['x_src'] = self.x_src[item,:,:]
        data['y_src'] = self.y_src[item,:,:]
        data['x_trgt'] = self.x_trgt[item,:,:]
        data['y_trgt'] = self.y_trgt[item,:,:]

        return data
    def prepare_src_trgt_data(self, src_idx, trgt_idx,window,stride,dof):
        x_src = get_robo_windows(self.data[dof],src_idx,window,stride)
        y_src =get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1),src_idx,window,stride)

        x_trgt = get_robo_windows(self.data[dof], trgt_idx, window, stride)
        y_trgt = get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1), trgt_idx, window, stride)
        return x_src,y_src,x_trgt,y_trgt

    def get_src_trgt_indices(self,):
        idx_b = np.where(self.data['robot_type'] == 0)[0]
        idx_c = np.where(self.data['robot_type'] == 1)[0]
        return idx_b,idx_c

'''
class LeggedRobotsDataset_semisup_DA(Dataset):
    'Data augmentation semi sup'
    def __init__(self, path,src_list,trg_list,window=50,stride=10,dof ='vel',replace_labels_file=-1):
        self.path = path
        self.data= self.load()
        self.dof = dof
        self.normalize()
        self.x_src,self.y_src,self.x_trgt, self.y_trgt = self.prepare_src_trgt_data(src_idx=src_list,trgt_idx=trg_list,window=window,stride=stride,dof = dof)
        self.no_classes = len(np.unique(self.y_src))
    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        #class_names = data.pop('class_names')

        for key in ['dof_pos', 'dof_vel']:
            data[key] = data[key].astype(np.float32)

        return data
    def normalize(self):
        pos_mean, pos_std = self.data['dof_pos'].mean(axis=(0,2), keepdims=True), self.data['dof_pos'].std(axis=(0,2), keepdims=True)
        self.data['pos'] = (self.data['dof_pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['dof_vel'].mean(axis=(0, 2), keepdims=True), self.data['dof_vel'].std(axis=(0, 2), keepdims=True)
        self.data['vel'] = (self.data['dof_vel'] - vel_mean) / vel_std


    def __len__(self):
        len = min(self.x_src.shape[0],self.x_trgt.shape[0])

        return len

    def __getitem__(self, item):
        data = {}
        data['x_src'] = self.x_src[item,:,:]
        data['y_src'] = self.y_src[item,:,:]
        data['x_trgt'] = self.x_trgt[item,:,:]
        data['y_trgt'] = self.y_trgt[item,:,:]

        return data
    def prepare_src_trgt_data(self, src_idx, trgt_idx,window,stride,dof):
        x_src = get_robo_windows(self.data[dof],src_idx,window,stride)
        y_src =get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1),src_idx,window,stride)

        x_trgt = get_robo_windows(self.data[dof], trgt_idx, window, stride)
        y_trgt = get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1), trgt_idx, window, stride)
        return x_src,y_src,x_trgt,y_trgt'''



class LeggedRobotsDataset_semisup_DA(Dataset):
    'Data augmentation semi sup'
    def __init__(self, path,src_list,trg_list,window=50,stride=10,dof ='vel',replace_labels_file=-1):
        self.path = path
        self.data= self.load()
        self.dof = dof
        self.normalize()
        self.idx_b,self.idx_c = self.get_src_trgt_indices()
        self.x_src,self.y_src,self.x_trgt, self.y_trgt = self.prepare_src_trgt_data(src_idx=list(self.idx_b),trgt_idx=list(self.idx_c),window=window,stride=stride,dof = dof)
        self.no_classes = len(np.unique(self.y_src.reshape(-1,)))
    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        #class_names = data.pop('class_names')

        for key in ['pos', 'vel']:
            data[key] = data[key].astype(np.float32)

        return data

    def get_src_trgt_indices(self,):
        idx_b = np.where(self.data['robot_type'] == 0)[0]
        idx_c = np.where(self.data['robot_type'] == 1)[0]
        return idx_b,idx_c

    def normalize(self):
        pos_mean, pos_std = self.data['pos'].mean(axis=(0,2), keepdims=True), self.data['pos'].std(axis=(0,2), keepdims=True)
        self.data['pos'] = (self.data['pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['vel'].mean(axis=(0, 2), keepdims=True), self.data['vel'].std(axis=(0, 2), keepdims=True)
        self.data['vel'] = (self.data['vel'] - vel_mean) / vel_std
    '''
    def __getitem__(self, item):
        data = {key: x[item] for key, x in self.data.items()}
        return data

    def __len__(self):
        return self.data['pos'].shape[0]
    '''

    def __len__(self):
        len = min(self.x_src.shape[0],self.x_trgt.shape[0])

        return len

    def __getitem__(self, item):
        data = {}
        data['x_src'] = self.x_src[item,:,:]
        data['y_src'] = self.y_src[item,:,:]
        data['x_trgt'] = self.x_trgt[item,:,:]
        data['y_trgt'] = self.y_trgt[item,:,:]

        return data
    def prepare_src_trgt_data(self, src_idx, trgt_idx,window,stride,dof):
        x_src = get_robo_windows(self.data[dof],src_idx,window,stride)
        y_src =get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1),src_idx,window,stride)

        x_trgt = get_robo_windows(self.data[dof], trgt_idx, window, stride)
        y_trgt = get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1), trgt_idx, window, stride)
        return x_src,y_src,x_trgt,y_trgt



def get_robo_windows(dataset, rob_idx, window=600, stride=5):

    'This function creates a numpy array for rob_idx: rob idx for training'
    windowed_robs = []
    windowed_robs = np.asarray(windowed_robs)

    dataset = dataset.swapaxes(1, 2)
    windowed_robs = torch.from_numpy(dataset[rob_idx,:,:])

    if window != -1:
        windowed_robs = windowed_robs.unfold(1, window, stride)
        windowed_robs = windowed_robs.reshape(-1, dataset.shape[-1], window).transpose(1,2).cpu().numpy()
    else:
        windowed_robs = windowed_robs.cpu().numpy()

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
    return windowed_robs

def sliding_window(a, win_size,stride_step):
    '''Slding window view of a 2D array a using numpy stride tricks.
        For a given input array `a` and the output array `b`, we will have
        `b[i] = a[i:i+w]`

        Args:
            a: numpy array of shape (N,M)
        Returns:
            numpy array of shape (K,w,M) where K=N-w+1
        '''


    shape = (a.shape[0] - win_size + 1, win_size) + a.shape[-1:]
    strides = (a.strides[0],) + a.strides
    unstrided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    stride_samples = np.linspace(0, unstrided.shape[0], num=int(unstrided.shape[0]/stride_step)).astype(int)
    return unstrided[stride_samples[:-1],:,:]


class LeggedRobotsDataset_semisup(Dataset):
    def __init__(self, data,label,robo_labels,frac_labels,labels_p_class =10):
        self.data = data
        self.label = label

        #number of labels for the ssl setting
        self.tot_labels = self.data.shape[0]
        self.frac_labels = frac_labels
        self.no_labels = int(self.frac_labels*self.tot_labels)
        #self.no_labels = labels_p_class
        self.rob_type_labels = robo_labels
        print()
        print("here")
        temp = np.mean(self.label,axis=1)
        temp_rob_type = np.mean(self.rob_type_labels ,axis=1)
        self.labels_unique = list(np.unique(np.round(temp)))
        self.rob_a= np.where(temp_rob_type ==0)[0]
        self.rob_b = np.where(temp_rob_type ==1)[0]
        self.idx_0 = np.where(temp==0)[0]
        self.idx_1 = np.where(temp == 1)[0]
        self.idx_2 = np.where(temp == 2)[0]
        self.idx_3 = np.where(temp == 3)[0]
        self.idx_4 = np.where(temp == 4)[0]

        self.idx_0_a = np.intersect1d(self.idx_0 ,self.rob_a )
        self.idx_1_a = np.intersect1d(self.idx_1, self.rob_a)
        self.idx_2_a = np.intersect1d(self.idx_2, self.rob_a)
        self.idx_3_a = np.intersect1d(self.idx_3, self.rob_a)
        self.idx_4_a = np.intersect1d(self.idx_4, self.rob_a)

        self.idx_0_b = np.intersect1d(self.idx_0, self.rob_b)
        self.idx_1_b = np.intersect1d(self.idx_1, self.rob_b)
        self.idx_2_b = np.intersect1d(self.idx_2, self.rob_b)
        self.idx_3_b = np.intersect1d(self.idx_3, self.rob_b)
        self.idx_4_b = np.intersect1d(self.idx_4, self.rob_b)


    def __len__(self):
        return self.no_labels
    def __getitem__(self, item):
        'chose random numb'
        idx_lbl = np.random.randint(5, size=1).item()

        if idx_lbl == 0:

            idx_f =  self.data[np.random.choice(self.idx_0_a,size=1).item(),: ,:]
            idx_s = self.data[np.random.choice(self.idx_0_b,size=1).item(),: ,:]
        elif idx_lbl == 1:
            idx_f = self.data[np.random.choice(self.idx_1_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_1_b,size=1).item(), :, :]
        elif idx_lbl == 2:
            idx_f = self.data[np.random.choice(self.idx_2_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_2_b,size=1).item(), :, :]
        elif idx_lbl == 3:
            idx_f = self.data[np.random.choice(self.idx_3_a,size=1).item(), :, :]
            try:
                idx = np.random.choice(self.idx_3_b,size=1).item()
                idx_s = self.data[idx,:, :]
            except(ValueError):
                print("here value")
        elif idx_lbl == 4:
            idx_f = self.data[np.random.choice(self.idx_4_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_4_b,size=1).item(), :, :]
        data = {}
        data['x_1'] = idx_f
        data['x_2'] = idx_s
        data['idx_lbl'] = idx_lbl


        return data
    def prepare_src_trgt_data(self, src_idx, trgt_idx,window,stride):
        x_src = get_robo_windows(self.data['vel'],src_idx,window,stride)
        y_src =get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1),src_idx,window,stride)

        x_trgt = get_robo_windows(self.data['vel'], trgt_idx, window, stride)
        y_trgt = get_robo_windows(np.expand_dims(self.data['terrain_type'],axis=1), trgt_idx, window, stride)
        return x_src,y_src,x_trgt,y_trgt

class LeggedRobotsDatasetwind_DA_semisup(Dataset):
    'dataloader for DA that takes windows.. basically a window for sorc and target. (multiple for src) and target if avlbl and hierarchical contrastive loss'
    def __init__(self, path,src_list,trgt_list, window,stride,dof,frac_labels,labels_p_class =10):

        self.path = path
        self.data, self.class_names = self.load()
        self.dof = dof
        self.normalize()
        self.x_src, self.y_src, self.x_trgt, self.y_trgt = self.prepare_src_trgt_data(src_idx=src_list,
                                                                                      trgt_idx=trgt_list, window=window,
                                                                                      stride=stride, dof=dof)
        self.no_classes = len(np.unique(self.y_src))


        #number of labels for the ssl setting
        self.tot_labels = self.y_trgt.shape[0]
        self.frac_labels = frac_labels
        self.no_label_target = int(self.frac_labels*self.tot_labels)
        #self.no_labels = labels_p_class
        self.labels_trgt = np.mean(self.y_trgt)
        self.labels_src = np.round(np.mean(self.y_src,axis=1)).astype(int).reshape(-1,)
        self.labels_trgt = np.round(np.mean(self.y_trgt, axis=1)).astype(int).reshape(-1, )


       
        self.idx_0_s = np.where( self.labels_src==0) [0]
        self.idx_1_s = np.where(self.labels_src==1)[0]
        self.idx_2_s = np.where(self.labels_src==2)[0]
        self.idx_3_s = np.where(self.labels_src==3)[0]
        self.idx_4_s = np.where(self.labels_src==4)[0]

        self.idx_0_t = np.where(self.labels_trgt == 0)[0]
        self.idx_1_t = np.where(self.labels_trgt == 1)[0]
        self.idx_2_t = np.where(self.labels_trgt == 2)[0]
        self.idx_3_t = np.where(self.labels_trgt == 3)[0]
        self.idx_4_t = np.where(self.labels_trgt == 4)[0]
        self.labels_trgt_pseudo = -1*np.ones(self.labels_trgt.shape[0])

        lbl_idx = np.random.choice(np.arange(0,self.labels_trgt.shape[0]), size=self.no_label_target  )
        self.labels_trgt_pseudo[lbl_idx] = self.labels_trgt[lbl_idx]

    def __len__(self):
        return min(self.labels_src.shape[0],self.labels_trgt.shape[0])
    def __getitem__(self, item):
        'chose random numb'

        data = {}
        data['x_1_src'] = self.x_src[item,:,:]
        src_label = self.labels_src[item].astype(int)
        data['y_src'] = src_label



        item_idx = np.random.choice(eval(f'self.idx_{str(src_label)}_s'),size=1)[0]
        data['x_2_src'] = self.x_src[item_idx, : ,:]
        data['x_1_trgt'] = self.x_trgt[item,:,:]
        data['y_trgt'] = self.labels_trgt_pseudo[item]
        trgt_label= self.labels_trgt_pseudo[item].astype(int)
        if trgt_label == -1:
            data['x_2_trgt'] =  self.x_trgt[item,:,:]
        else:
            idx = np.random.choice(eval(f'self.idx_{str(trgt_label)}_t'), size=1)[0]
            data['x_2_trgt'] = self.x_trgt[idx, :, :]

        return data
        '''
        code to do uniform target sampling
        idx_lbl = np.random.randint(5, size=1).item()
            
        if idx_lbl == 0:

            idx_f =  self.data[np.random.choice(self.idx_0_a,size=1).item(),: ,:]
            idx_s = self.data[np.random.choice(self.idx_0_b,size=1).item(),: ,:]
        elif idx_lbl == 1:
            idx_f = self.data[np.random.choice(self.idx_1_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_1_b,size=1).item(), :, :]
        elif idx_lbl == 2:
            idx_f = self.data[np.random.choice(self.idx_2_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_2_b,size=1).item(), :, :]
        elif idx_lbl == 3:
            idx_f = self.data[np.random.choice(self.idx_3_a,size=1).item(), :, :]
            try:
                idx = np.random.choice(self.idx_3_b,size=1).item()
                idx_s = self.data[idx,:, :]
            except(ValueError):
                print("here value")
        elif idx_lbl == 4:
            idx_f = self.data[np.random.choice(self.idx_4_a,size=1).item(), :, :]
            idx_s = self.data[np.random.choice(self.idx_4_b,size=1).item(), :, :]
        


        data['idx_lbl'] = idx_lbl
        '''

        return data

    def load(self):
        data = np.load(self.path, allow_pickle=True).item()
        class_names = data.pop('class_names')

        for key in ['pos', 'vel']:
            data[key] = data[key].astype(np.float32)

        return data, class_names

    def normalize(self):
        pos_mean, pos_std = self.data['pos'].mean(axis=(0, 2), keepdims=True), self.data['pos'].std(axis=(0, 2),
                                                                                                    keepdims=True)
        self.data['pos'] = (self.data['pos'] - pos_mean) / pos_std

        vel_mean, vel_std = self.data['vel'].mean(axis=(0, 2), keepdims=True), self.data['vel'].std(axis=(0, 2),
                                                                                                    keepdims=True)
        self.data['vel'] = (self.data['vel'] - vel_mean) / vel_std

    def prepare_src_trgt_data(self, src_idx, trgt_idx, window, stride, dof):
        x_src = get_robo_windows(self.data[dof], src_idx, window, stride)
        y_src = get_robo_windows(np.expand_dims(self.data['terrain_type'], axis=1), src_idx, window, stride)

        x_trgt = get_robo_windows(self.data[dof], trgt_idx, window, stride)
        y_trgt = get_robo_windows(np.expand_dims(self.data['terrain_type'], axis=1), trgt_idx, window, stride)
        return x_src, y_src, x_trgt, y_trgt
def get_segments_across_changes(X,Y_labels,win_length,buff=5):
    print(X.shape)
    diff_labl = np.concatenate((np.zeros((X.shape[0],1)),np.abs(np.diff(Y_labels))),axis=1)
    cp_indices_all = (diff_labl != 0).astype(int)


    '''window block: A big window size before and after the cp from where smaller segments of sim
    dissim pairs are extracted. (should be large enough to get 2 pairs)
    pair_length: length of smaller segments extracted from larger window black

    '''
    data_list = []

    for idx in range(0,X.shape[0]):
        signal = X[idx,:,:]
        cp_indices = np.where(cp_indices_all[idx,:] == 1)[0]
        lbl = Y_labels[idx,:]


        # seg_size has to be even

        'window block determined from consecutive cps but could be changed later'



        for cp in cp_indices:  ## For non yahoos cp_indices[:-2]:
            data = {}
            'Checking if sufficient length available for last cp to get segment after cp'
            X_p = signal[cp - win_length - buff: cp - buff, :]

            X_f = signal[cp + buff: cp + win_length + buff, :]
            if len(X_f) < win_length or len(X_p) < win_length:
                continue
            X_p_lbl = int(np.mean(lbl[cp - win_length - buff: cp - buff]))
            X_f_lbl = np.mean(lbl[cp + buff: cp + win_length + buff])
            data['X_p'] = X_p
            data['X_f'] = X_f
            data['X_p_lbl'] = X_p_lbl
            data['X_f_lbl'] = X_f_lbl
            data_list.append(data)




    return data_list


    return data_list


class SimDisimLoader(Dataset):
    def __init__(self,paired_list1,paired_list2):
        self.paired_list1 = paired_list1
        self.paired_list2 = paired_list2

        self.labels_unique1 = np.asarray([dct['X_p_lbl'] for dct in self.paired_list1] )
        self.labels_unique2 = np.asarray([dct['X_p_lbl'] for dct in self.paired_list2] )

        self.idx_a={}
        self.idx_a['0'] = np.where(self.labels_unique1 == 0)[0]
        self.idx_a['1'] = np.where(self.labels_unique1== 1)[0]
        self.idx_a['2'] = np.where(self.labels_unique1== 2)[0]
        self.idx_a['3'] = np.where(self.labels_unique1== 3)[0]
        self.idx_a['4'] = np.where(self.labels_unique1== 4)[0]

        self.idx_b = {}
        self.idx_b['0'] = np.where(self.labels_unique2 == 0)[0]
        self.idx_b['1'] = np.where(self.labels_unique2 == 1)[0]
        self.idx_b['2']= np.where(self.labels_unique2 == 2)[0]
        self.idx_b['3'] = np.where(self.labels_unique2 == 3)[0]
        self.idx_b['4'] = np.where(self.labels_unique2 == 4)[0]

    def __len__(self):
        return len(self.labels_unique1)

    def __getitem__(self, item):
        'main idea . Get past and future window lists. Look at first label.. Check what corresponds to this label in the other list'
        X_p =self.paired_list1[item]['X_p']
        X_f = self.paired_list1[item]['X_f']
        data= {}
        data['X_p_d1'] = X_p
        data['X_f_d1'] = X_f
        lbl = self.paired_list1[item]['X_p_lbl']
        data['X_p_d_d2'] = X_p
        idx_f = self.paired_list2[np.random.choice(self.idx_b[str(lbl)], size=1)[0]]['X_p']
        data['X_p_d_d2'] = idx_f
        return data