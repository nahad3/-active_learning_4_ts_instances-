'Taking a custom take on https://arxiv.org/pdf/2111.14834.pdf'

import torch
import numpy as np
from active_learning_loop import train_active_loop
from torch.utils.data import DataLoader
from models_TCN.advrs_DA import classifier_model
from models_TCN import  TSEncoder
from utils_vis import   plot_ts_reps

import os
from sklearn.model_selection import train_test_split
from utils.robo_dataloader_bams import LeggedRobotsDataset_DA,get_robo_windows,LeggedRobotsDataset_semisup_DA

import argparse

import wandb
import os

parser = argparse.ArgumentParser()


'visualize data'
'Goal is to only pass on fraction of labels per class'
parser.add_argument('--gpu',type=int,default=4,help='gpu for device')
parser.add_argument('--src_dom',type=str,default='B',help='Source domain (B or C)')
parser.add_argument('--trgt_dom',type=str,default='C',help='Source domain (B or C)')
parser.add_argument('--load', type=int,default=0, help='load DA model')
parser.add_argument('--model_pth',type=str,default='saved_models_ssl/ssl_save_robots_no_fixed',help='file path to saved model')
parser.add_argument('--test', type=int,default=0, help='test or not')
parser.add_argument('--total_budget',type=int,default = 100,help="total number of training points (total pool at the end of all training)")
parser.add_argument('--no_queries', type=int,default=100, help='No of queries to add to pool after each training session')
parser.add_argument('--query_type',type=int,default=0,help='{0: for random queriy 1: for max entropy}')
parser.add_argument('--train', type=int,default=1, help='train or not')
parser.add_argument('--batch_size', type=int,default=12, help='train or not')
parser.add_argument('--visualize', type=int,default=0, help='train or not')
parser.add_argument('--dof', type=str,default='pos', help='do DANN on top of ssl reps')
parser.add_argument('--backbone_path',type=str,default = './saved_models_ssl/robo_tcn/robo_TS2Vec_bbone_new')
parser.add_argument('--dataset_type',type=str,default = 'small')
parser.add_argument('--weight_target_sup',type=float,default = 1)
parser.add_argument('--lr_src',type=float,default = 0.0001)
parser.add_argument('--no_epochs',type=int,default = 401)
parser.add_argument('--file_path_model_base',type=str,default='./saved_models/robo_tcn')
parser.add_argument('--window',type=float,default=1500,help='default load')
parser.add_argument('--stride',type=float,default=1500,help='default load')


args = parser.parse_args()
device =  torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
dof = args.dof
rob_no = 1
dof_idx = np.arange(0,10)

input_dims = 12
output_dims = 8
hidden_dims = 66
depth = 5
enc_out_dims = 66
batch_size = args.batch_size
proj_dims =66

#path for trained SSL model

assert args.query_type == 0 or args.query_type == 1,  f"Query type should be either 0 for random or 1 for max entropy. Got: {args.query_type}"

if not os.path.exists(args.file_path_model_base):
    os.makedirs(args.file_path_model_base)



args.file_path_save_src_feats = os.path.join(args.file_path_model_base,'robo_src_feats_saved')
args.file_path_save_src_clfr = os.path.join(args.file_path_model_base,'robo_src_clfr_saved')

if args.dataset_type == 'small':
    data_path = 'updated_robot_data/legged_robots_v1.npy'
else:
    data_path ='updated_robot_data/robot_dataset_new_transf_saved_short.npy'
t_brd_path = 'runs/dom_dpt/'



if args.src_dom == 'B':
    string_sv_DA = 'B_to_C'
    src_idx =  list(np.arange(0,30))
    trgt_idx = list(np.arange(40,70))
elif args.src_dom == 'C':
    string_sv_DA = 'C_to_B'
    src_idx = list(np.arange(40,60 ))
    trgt_idx = list(np.arange(0, 20))






args_dict = vars(args)
args_dict['data_path'] = data_path

os.environ["WANDB_SILENT"] = "true"
wandb.init(config=args_dict)

if args.window == -1:
    stride = 1500
else:
    stride = args.window
dataset = LeggedRobotsDataset_semisup_DA(data_path,window=args.window,stride=stride,src_list=src_idx ,trg_list=trgt_idx,dof=dof)

no_classes = dataset.no_classes
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.3, random_state=42)

train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True,sampler=train_idx)
val_loader = DataLoader(dataset, batch_size=64, drop_last=False,sampler=test_idx)

temp = list(train_loader)
no_batches = len(temp)
len_per_batch  = (temp[0]['y_src'].reshape(-1,).cpu().numpy()).shape[0]



args.total_train_points = args.total_budget

args_dict = vars(args)
args_dict['data_path'] = data_path

os.environ["WANDB_SILENT"] = "true"
wandb.init(config=args_dict)
print(f"total_train_points : {args.total_train_points}")
"Defining models"
model_feats_src = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                      hidden_dims=hidden_dims, depth=depth).to(device)

model_src_clfr = classifier_model(input_dims=output_dims, no_classes=no_classes).to(device)


if args.load:
    checkpoint = torch.load(args.file_path_save_src_feats)
    model_feats_src.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load(args.file_path_save_src_clfr)
    model_src_clfr.load_state_dict(checkpoint['model_state_dict'])
if args.train:

    model_da = train_active_loop(model_feats=model_feats_src, device=device, args=args, model_clfr=model_src_clfr,
                          train_dataloader=train_loader,
                         wanb_con=wandb, val_dataload=val_loader)









rob_idx_c_test = [0,5,40,44]
i = 0

dataset_vis = LeggedRobotsDataset_DA(data_path,window= -1,stride=1500,src_list=rob_idx_c_test ,trg_list=rob_idx_c_test,dof = dof )
x = dataset_vis.data[dof][i,:,:].swapaxes(0,1)
y  = dataset_vis.data['terrain_type'][i,:]
plot_ts_reps(model_feats = model_feats_src,model_clfr = model_src_clfr,device=device,x = x, y = y,window =-1)