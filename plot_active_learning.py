'''Code for plottig active learning results'''
import os

import numpy as np
import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 30})





dict_dataset = {0:'vel_robot_B',1: 'vel_robot_C',2:'pos_robot_C',3:'pos_robot_B',4:'high_freq_sinusoid',5:'low_freq_sinusoid'}
'Chose corresponding dictionary number'

data_idx = 1

#data_type  = 'vel_robot_B'
data_type  = dict_dataset[data_idx]
base_path = f'./results/{data_type}'

'Choose the appropriate  result files in the list'

#file_list = ['result_InfoNN.npz','result_Experimental method.npz','result_Coreset.npz','result_max entropy.npz','result_random.npz']
#file_list = ['result4_Experimental method.npz', 'result4_Experimental method with marg entrp.npz',
#             'result4_random.npz','result4_InfoNN.npz','result4_max entropy.npz'  ,'result4_Coreset.npz',
#           ]


file_list = ['result4_Experimental method.npz',
             'result4_random.npz','result4_max entropy.npz'
           ]
#file_list = ['result4_max entropy.npz'  ,'result4_random.npz','result4_Experimental method.npz','result4_InfoNN.npz' ] #,'result4_Coreset.npz']
#file_list = ['result2_Experimental method.npz','result2_random.npz']
color_list = ['b','g','r','y','m','c'] # ','tab:red','tab:green']
fig, ax = plt.subplots()

path_load = os.path.join(base_path,file_list[0])
accuracy_array = np.load(path_load)

n_samples = accuracy_array['no_queries_round'].item()
#n_total =   220

#total points x times points added
n_total = accuracy_array['no_queries_round'].item()*accuracy_array['acc_list'].shape[-1]
num_trials = 5

x = np.arange(n_samples, n_total+1, n_samples)

for k in range(0,len(file_list)):
    path_load = os.path.join(base_path,file_list[k])
    accuracy_array = np.load(path_load)
    accuracy_array = accuracy_array['acc_list']
    accuracy_q1 = np.quantile(accuracy_array, 0.25, axis=0)
    accuracy_q3 = np.quantile(accuracy_array, 0.75, axis=0)
    accuracy_median = np.median(accuracy_array, axis=0)
    plt.plot(x, accuracy_median, c=color_list[k], linewidth=4)
    plt.fill_between(x, accuracy_q1, accuracy_q3, color=color_list[k], alpha=0.2)



# ax.legend(["Info-NN(with clustering)", "K-Center", "MaxEntropy", "Margin Sampling", "BatchBALD", "Random"], loc=4)
#leg = plt.legend(["Info-NN", "Experimental method", "Coreset", "Max k entropy","Random"], loc=4, prop={'size': 30})
leg = plt.legend(["Our Method w Entp", "Our Method w Mrg","Random","InfoNN","Max k entropy","Coreset" ]) # ,"Experimental method 2"], loc=4, prop={'size': 30})
#leg = plt.legend(["Experimental method max entropy" ])
leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)
#plt.xticks(np.arange(n_samples, n_total+1, 0))
plt.xlabel('Number of labels')
plt.ylabel('Classification Accuracy')
plt.title(data_type)
fig.set_size_inches(12, 9)
plt.savefig(f'./results/plots/{data_type}2.png', dpi=100, bbox_inches='tight')