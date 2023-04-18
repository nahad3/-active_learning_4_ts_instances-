'''Code for plottig active learning results'''

import numpy as np
import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 30})

n_samples = 20
n_total = 100
num_trials = 5
x = np.arange(n_samples, n_total+1, n_samples)
# x = range(0, n_total, n_samples)

file_list = ['./results/result_Experimental method.npz','./results/result_random.npz','./results/result_max entropy.npz']
color_list = ['b','g','r'] # ','tab:red','tab:green']
fig, ax = plt.subplots()

for k in range(0,len(file_list)):
    accuracy_array = np.load(file_list[k])
    accuracy_array = accuracy_array['acc_list']
    accuracy_q1 = np.quantile(accuracy_array, 0.25, axis=0)
    accuracy_q3 = np.quantile(accuracy_array, 0.75, axis=0)
    accuracy_median = np.median(accuracy_array, axis=0)
    plt.plot(x, accuracy_median, c=color_list[k], linewidth=4)
    plt.fill_between(x, accuracy_q1, accuracy_q3, color=color_list[k], alpha=0.2)



# ax.legend(["Info-NN(with clustering)", "K-Center", "MaxEntropy", "Margin Sampling", "BatchBALD", "Random"], loc=4)
leg = plt.legend(["Experimental method",  "Random","Max k entropy"], loc=4, prop={'size': 30})
leg.get_frame().set_facecolor('none')
leg.get_frame().set_linewidth(0.0)
plt.xticks(x)
plt.xlabel('Number of labels')
plt.ylabel('Classification Accuracy')

fig.set_size_inches(12, 9)
plt.savefig('./results/plots/experinment1.png', dpi=100, bbox_inches='tight')