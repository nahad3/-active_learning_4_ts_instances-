import torch
import numpy as np
from torch.utils.data import Dataset
from scipy import signal

def get_list_sinusoids(Fs,freq_list, label_dict, saw_tooth=0,noise=0.7):
    '''this function generates a siniusioid that starts from a base freqency and doubles it succesivbe time
    FS: samples for one period. label dict is for converitng frequences to labels'''
    x = np.arange(Fs)
    samples = Fs
    x_array = []
    x_array = np.asarray(x_array)
    y_labels = np.array(x_array)
    noise = noise

    for i in range(0,len(freq_list)):
        if saw_tooth == 1:
            x_temp = signal.sawtooth(2 * (freq_list[i]) * np.pi * x / Fs).reshape(-1, )
        else:
            x_temp = np.sin(2 * (freq_list[i]) * np.pi * x / Fs).reshape(-1, )
        x_array = np.concatenate((x_array, x_temp), axis=0) if len(x_array) else x_temp
        labels = label_dict[freq_list[i]] * np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
    x_array = x_array + np.random.normal(0, noise, x_array.shape[0])
    idx =  np.random.randint(-samples, samples, size=1)[0]
    y_labels = np.roll(y_labels, idx)
    x_array = np.roll(x_array, idx)
    return x_array,y_labels


def get_list_gaussians(Fs,mu_list,var_list):
    x = np.arange(Fs)
    samples = Fs
    x_array = []
    x_array = np.asarray(x_array)
    y_labels = np.array(x_array)
    noise = 0.7
    for i in range(0,len(mu_list)):

        x_temp = np.random.normal(loc=mu_list[i], scale=var_list[i], size=Fs)
        x_array = np.concatenate((x_array, x_temp), axis=0) if len(x_array) else x_temp
        labels = i * np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
    #x_array = x_array + np.random.normal(0, noise, x_array.shape[0])
    idx =  np.random.randint(-samples, samples, size=1)[0]
    y_labels = np.roll(y_labels, idx)
    x_array = np.roll(x_array, idx)
    return x_array,y_labels

class window_sampler(Dataset):
    'creates a dataloader  over windows'
    def __init__(self,dataset):
        super(window_sampler, self).__init__()
        self.dataset = dataset
    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self, index):
        return self.dataset[index,:,:]

class generic_sampler(Dataset):
    'creates a dataloader  for generic windows (Based on robot dataset'
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        #super(window_sampler, self).__init__()
        #self.dataset = dataset
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        data = {}
        data['x'] = self.X[index, :, :]
        data['y'] = self.Y[index, :]
        return data

class tsk_btch_sampler(Dataset):
    def __init__(self,dataset):
        super(tsk_btch_sampler, self).__init__()
        self.dataset = dataset
    def __len__(self):
        return self.dataset[0].shape[1]
    def __getitem__(self, index):
        #Created tuple for tasks whhere task information is important
        return (self.dataset[0][:,index,:,:],self.dataset[1])
def gen_sinusoid( base_freq, change=1, samples=8000, Fs=4000, no_changes=10, amp_chage=0,sawtooth=0):
    # Fs: Number of samples used to generate one period when Freq is 1Hz
    # no of samples when Fs=4000 (no of samples of one period
    x = np.arange(samples)
    y_temp = []
    y_array = np.asarray(y_temp)
    y_labels = np.array(y_temp)
    noise = 0.7
    for i in range(no_changes):
        if amp_chage == 1:
            x_temp = 2 * np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        elif change == 1:
            #remove 2 below
            if sawtooth == 0:
                x_temp = np.sin(2 * (1 + (((i) % 2))) * np.pi * base_freq * x / Fs).reshape(-1, )
            else:
                x_temp = signal.sawtooth(2 * (1 + (((i) % 2))) * np.pi * base_freq * x / Fs).reshape(-1, )
        else:
            x_temp = np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        y_array = np.concatenate((y_array, x_temp), axis=0) if len(y_array) else x_temp
        labels = ((i) % 2) * np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
    y_array = y_array + np.random.normal(0, noise, y_array.shape[0])
    idx = np.max([0, np.random.randint(-samples, samples, size=1)[0]])
    y_labels = np.roll(y_labels, idx)
    y_array = np.roll(y_array, idx)
    return y_array, y_labels

def gen_sinusoid_swtch_sawtooth( base_freq, change=1, samples=8000, Fs=4000, no_changes=10, amp_chage=0):
    # Fs: Number of samples used to generate one period when Freq is 1Hz
    # no of samples when Fs=4000 (no of samples of one period
    x = np.arange(samples)
    y_temp = []
    y_array = np.asarray(y_temp)
    y_labels = np.array(y_temp)
    noise = 0.7
    for i in range(no_changes):
        if amp_chage == 1:
            x_temp = 2 * np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        elif change == 1:
            #remove 2 below
            if i % 2 ==0:
                x_temp = np.sin(2 * (1 + (((i) % 2))) * np.pi * base_freq * x / Fs).reshape(-1, )
            else:
                x_temp = signal.sawtooth(2 * (1 + (((i) % 2))) * np.pi * base_freq * x / Fs).reshape(-1, )
        else:
            x_temp = np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        y_array = np.concatenate((y_array, x_temp), axis=0) if len(y_array) else x_temp
        labels = ((i) % 2) * np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
    y_array = y_array + np.random.normal(0, noise, y_array.shape[0])
    idx = np.max([0, np.random.randint(-samples, samples, size=1)[0]])
    y_labels = np.roll(y_labels, idx)
    y_array = np.roll(y_array, idx)
    return y_array, y_labels
def gen_sinusoid( base_freq, change=1, samples=8000, Fs=4000, no_changes=10, amp_chage=0):
    # Fs: Number of samples used to generate one period when Freq is 1Hz
    # no of samples when Fs=4000 (no of samples of one period
    x = np.arange(samples)
    y_temp = []
    y_array = np.asarray(y_temp)
    y_labels = np.array(y_temp)
    noise = 0.7
    for i in range(no_changes):
        if amp_chage == 1:
            x_temp = 2 * np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        elif change == 1:
            #remove 2 below
            x_temp = np.sin(2 * (1 + (((i) % 2))) * np.pi * base_freq * x / Fs).reshape(-1, )
        else:
            x_temp = np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
        y_array = np.concatenate((y_array, x_temp), axis=0) if len(y_array) else x_temp
        labels = ((i) % 2) * np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
    y_array = y_array + np.random.normal(0, noise, y_array.shape[0])
    idx = np.max([0, np.random.randint(-samples, samples, size=1)[0]])
    y_labels = np.roll(y_labels, idx)
    y_array = np.roll(y_array, idx)
    return y_array, y_labels

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

class Multi_modal_synth_reg_data(Dataset):
    def __init__(self,n_batch=200,n_shots=10,n_query = 10):
        super(Multi_modal_synth_reg_data, self).__init__()
        self.n_batch  = n_batch
        self.n_shots = n_shots
        self.n_query = n_query
        self.n_episodes = 4
        self.n_dim = 1

    def __len__(self):
        return self.n_batch * self.n_episodes
    def __getitem__(self, index):
        a = np.random.uniform(0.1, 5, 1)
        phase = np.random.uniform(0,np.pi,1)

        x_shots = np.random.uniform(-5,5 ,(self.n_shots,self.n_dim))
        y_shots = a*np.sin((x_shots)+phase)
        x_query =  np.random.uniform(-5, 5,(self.n_query,self.n_dim))
        y_query = a*np.sin((x_query)+phase)

        return x_shots,y_shots,x_query,y_query

class Gen_Freq_switch_data(Dataset):
    def __init__(self,n_episodes= 1000,n_tasks=10):
        super(Gen_Freq_switch_data, self).__init__()
        self.n_episodes  = n_episodes
        self.n_tasks = n_tasks

    def gen_sinusoid(self,base_freq, change=1, samples=8000, Fs=4000, no_changes=10, amp_chage=0):
        # Fs: Number of samples used to generate one period when Freq is 1Hz
        # no of samples when Fs=4000 (no of samples of one period
        x = np.arange(samples)
        y_temp = []
        y_array = np.asarray(y_temp)
        y_labels = np.array(y_temp)
        noise = 0.7
        for i in range(no_changes):
            if amp_chage == 1:
                x_temp = 2 * np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
            elif change == 1:
                x_temp = np.sin(2 * (1 + (i) % 2) * np.pi * base_freq * x / Fs).reshape(-1, )
            else:
                x_temp = np.sin(2 * np.pi * base_freq * x / Fs).reshape(-1, )
            y_array = np.concatenate((y_array, x_temp), axis=0) if len(y_array) else x_temp
            labels = ((i) % 2) * np.ones(len(x_temp))
            y_labels = np.concatenate((y_labels, labels), axis=0) if len(y_labels) else labels
        y_array = y_array + np.random.normal(0, noise, y_array.shape[0])
        idx = np.max([0, np.random.randint(-samples, samples, size=1)[0]])
        y_labels = np.roll(y_labels, idx)
        y_array = np.roll(y_array, idx)
        return y_array, y_labels

    def __len__(self):
        return self.n_episodes * self.n_tasks
    def __getitem__(self, index):
        'get item would be a task consisting of getting a support set and query set'
        total_length = 6000
        a = np.random.uniform(0.1, 5, 1)
        phase = np.random.uniform(0,np.pi,1)
        #base_freq = np.random.choice(np.arange(8,16,2),1)[0]
        change = 1
        base_freq = np.random.choice(np.arange(8, 16, 2), 1)[0]
        #FS should be kept fixed
        #Fs = np.random.choice(np.arange(200,2000,500),1)[0]
        Fs = 500
        no_changes = int(total_length/Fs)
        task_type = np.random.choice(np.arange(0, 2, 1), 1)[0]
        if task_type == 0:
            x_series = self.gen_sinusoid(base_freq = base_freq,change = change, samples=Fs, Fs=Fs, no_changes=20)[0]
            mean = np.nanmean(x_series)
            std = np.nanstd(x_series)
            x_series = (x_series - mean) / std
            x_series = x_series.reshape(-1,1)
            window = 200
            stride = 10
            x_windowed = sliding_window(x_series, window, stride)
        elif task_type == 1:
            x_series = self.gen_sinusoid(base_freq=base_freq, change=change, samples=Fs, Fs=Fs, no_changes=20)[0]
            mean = np.nanmean(x_series)
            std = np.nanstd(x_series)
            x_series = (x_series - mean) / std
            x_series = x_series.reshape(-1, 1)
            window = 200
            stride = 10
            x_windowed = sliding_window(x_series, window, stride)
        tuple_w_task_type = (x_windowed,task_type)
        return tuple_w_task_type



class Robot_data(Dataset):
    def __init__(self,file_path,file_path_labels,n_episodes=100,n_tasks=16,task_types='both',window=100,stride= 5):
        '''data loader for Robot dataset
        inputs:
        file path and file labels path for the robots
        n_episodes: Number of outerloops
        n_tasks: number of tasks for an inner loop (Total length of dataset is n_episodes*n_tasks)
        task_types: what kind of tasks should be sampled. Both dof pos and dof vel or just one
        '''
        super(Robot_data,self).__init__()
        self.dof_pos = np.load(file_path)['dof_pos_array']
        self.dof_vel = np.load(file_path)['dof_vel_array']
        self.dof_pos = ( self.dof_pos - np.mean(self.dof_pos, axis = 0))/np.std(self.dof_pos,axis = 0)
        self.dof_vel = (self.dof_vel - np.mean(self.dof_vel, axis=0)) / np.std(self.dof_vel, axis=0)
        self.labl_coars = np.load(file_path_labels)['label_array_coarser']
        self.n_episodes = n_episodes
        self.n_tasks = n_tasks
        self.no_robs = self.dof_pos.shape[1]
        self.no_dof = self.dof_pos.shape[2]
        self.task_types = task_types
        self.window = window
        self.stride = stride

    def __len__(self):
        return self.n_episodes*self.n_tasks

    def __getitem__(self, item):
        'taking a task item to be one of random'
        'Leaving out alternate dofs and robot indices for held out tasks...'
        robot_idx =  np.random.choice(np.arange(0, self.no_robs, 2), 1)[0]
        #dof_idx =  np.random.choice(np.arange(0, self.no_dof, 2), 1)[0]
        dof_idx = 0
        if self.task_types == 'both':
            if np.random.rand(1)[0] >= 0.5:
                'if tasks sample of both type, then flip a coin to decide if pos sampled or vel'
                data_array = self.dof_pos[:,robot_idx,dof_idx]
            else:
                data_array = self.dof_vel[:, robot_idx, dof_idx]
        elif self.task_types == 'pos':
            data_array = self.dof_pos[:, robot_idx, dof_idx]
        elif self.task_types == 'vel':
            data_array = self.dof_vel[:, robot_idx, dof_idx]
        x_windowed = sliding_window(data_array.reshape(-1,1), self.window, self.stride)

        return x_windowed
def get_freq_mixture(Fs,freq_mixture_list,amp_list):
    x = np.arange(Fs)
    x_temp = np.zeros(Fs)
    for i in range(0,len(freq_mixture_list)):
        x_temp = x_temp + amp_list[i]*np.sin(2 * (freq_mixture_list[i]) * np.pi * x / Fs).reshape(-1, )
    return x_temp

def gen_mixtures_sinusoid( Fs = 4000,no_changes=10):
    #Fs: Number of samples used to generate one period when Freq is 1Hz
    #no of samples when Fs=4000
    samples = Fs
    y_temp = []
    y_array = np.asarray(y_temp )
    y_labels = np.array(y_temp)
    noise = 0.7
    for i in range(no_changes):
        ampl_list = np.random.choice(np.arange(1,5),3)
        freq_list = np.random.choice(np.arange(1, 40), 3)
        x_temp = get_freq_mixture(Fs,freq_list,ampl_list)
        y_array = np.concatenate((y_array,x_temp),axis=0) if len(y_array) else x_temp
        labels = i*np.ones(len(x_temp))
        y_labels = np.concatenate((y_labels,labels),axis=0) if len(y_labels) else labels
    y_array = y_array + np.random.normal(0, noise, y_array.shape[0])
    idx = np.max([0,np.random.randint(-samples, samples ,size=1)[0] ] )
    y_labels = np.roll(y_labels,idx)
    y_array = np.roll(y_array,idx)
    return y_array,y_labels


class RoboPretrainLoader(Dataset):
    def __init__(self, robo_data, robo_labels,  window=100,
                 stride=5):
        super(RoboPretrainLoader, self).__init__()
        self.robot_data = robo_data
        self.robo_labels = robo_labels
        rob_indices = np.arrange(0,12,2)
        dof_indices = np.arange(0,12,2)

