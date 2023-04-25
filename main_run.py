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
from datetime import datetime

parser = argparse.ArgumentParser()



parser.add_argument('--no_runs',type=int,default=5,help='gpu for device')
parser.add_argument('--gpu',type=int,default=0,help='gpu for device')
parser.add_argument('--src_dom',type=str,default='C',help='Source domain (B or C)')
parser.add_argument('--trgt_dom',type=str,default='B',help='Source domain (B or C)')
parser.add_argument('--load', type=int,default=0, help='load DA model')
parser.add_argument('--save_path_results', type=str,default='./results/', help='path storing results')
parser.add_argument('--model_pth',type=str,default='saved_models_ssl/ssl_save_robots_no_fixed',help='file path to saved model')
parser.add_argument('--test', type=int,default=0, help='test or not')
parser.add_argument('--total_budget',type=int,default = 401,help="total number of training points (total pool at the end of all training)")
parser.add_argument('--no_queries', type=int,default=20, help='No of queries to add to pool after each training round')
parser.add_argument('--query_type',type=int,default=4,help='{0: for random queriy 1: for max entropy 2: for entropy frequency,3: "InfoNN",4: "Coreset"} ')
parser.add_argument('--train', type=int,default=1, help='train or not')
parser.add_argument('--batch_size', type=int,default=12, help='train or not')
parser.add_argument('--visualize', type=int,default=0, help='train or not')
parser.add_argument('--dof', type=str,default='vel', help='type of dataset')
parser.add_argument('--backbone_path',type=str,default = './saved_models_ssl/robo_tcn/robo_TS2Vec_bbone_new')
parser.add_argument('--dataset_type',type=str,default = 'small')
parser.add_argument('--weight_target_sup',type=float,default = 1)
parser.add_argument('--lr_src',type=float,default = 0.0001)
parser.add_argument('--no_epochs',type=int,default = 601)
parser.add_argument('--file_path_model_base',type=str,default='./saved_models/robo_tcn')
parser.add_argument('--window',type=float,default=1500,help='default load')
parser.add_argument('--stride',type=float,default=1500,help='default load')
parser.add_argument('--path_pre_active_model',type=str,default='./saved_models/first_round_models',help='path to save init models')
parser.add_argument('--train_pre_acq_model',type=bool,default=True,help='Train model before any active pooling.')


args = parser.parse_args()
args.dataset = f'{args.dof}_robot_{args.src_dom}'
if not os.path.exists(args.save_path_results):
    os.makedirs(args.save_path_results)

args.save_path_results = os.path.join(args.save_path_results, args.dataset)
if not os.path.exists(args.save_path_results):
    os.makedirs(args.save_path_results)

args_dict = vars(args)
os.environ["WANDB_SILENT"] = "true"
project = 'Active Learning 4 TS'
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
name = args.dataset+'_'+dt_string
wandb.init(config=args_dict, project=project,name=name)
for q in range(0,5):
    args.query_type = q
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



    assert args.query_type == 0 or args.query_type == 1 or args.query_type == 2 or args.query_type == 3 or args.query_type == 4,  \
        f"Query type should be either 0 for random or 1 for max entropy or 2 for custom method. Got: {args.query_type}"


    if not os.path.exists(args.file_path_model_base):
        os.makedirs(args.file_path_model_base)


    f1_list_runs = []
    acc_list_runs = []

    dict_type ={0: 'random',1: 'max entropy',2: "Experimental method",3: "InfoNN",4: "Coreset"}
    args.dict_type = dict_type
    full_path_save_results = os.path.join(args.save_path_results,f'result_{dict_type[args.query_type]}')

    args.file_path_save_src_feats = os.path.join(args.file_path_model_base, 'robo_src_feats_saved')
    args.file_path_save_src_clfr = os.path.join(args.file_path_model_base, 'robo_src_clfr_saved')

    if args.dataset_type == 'small':
        data_path = 'updated_robot_data/legged_robots_v1.npy'
    else:
        data_path = 'updated_robot_data/robot_dataset_new_transf_saved_short.npy'
    t_brd_path = 'runs/dom_dpt/'

    if args.src_dom == 'B':
        string_sv_DA = 'B_to_C'
        src_idx = list(np.arange(0, 30))
        trgt_idx = list(np.arange(40, 70))
    elif args.src_dom == 'C':
        string_sv_DA = 'C_to_B'
        src_idx = list(np.arange(40, 70))
        trgt_idx = list(np.arange(0, 30))

    args_dict = vars(args)
    args_dict['data_path'] = data_path

    os.environ["WANDB_SILENT"] = "true"




    if args.window == -1:
        stride = 1500
    else:
        stride = args.window
    dataset = LeggedRobotsDataset_semisup_DA(data_path, window=args.window, stride=stride, src_list=src_idx,
                                             trg_list=trgt_idx, dof=dof)

    no_classes = dataset.no_classes
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.3, random_state=42)

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=np.sort(train_idx))
    val_loader = DataLoader(dataset, batch_size=64, drop_last=False, sampler=test_idx)

    temp = list(train_loader)
    no_batches = len(temp)
    len_per_batch = (temp[0]['y_src'].reshape(-1, ).cpu().numpy()).shape[0]

    args.total_train_points = args.total_budget


    args_dict['data_path'] = data_path


    print(f"total_train_points : {args.total_train_points}")
    print(f"Query selection method: {dict_type[args.query_type]} ")
    "Defining models"
    model_feats = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                                hidden_dims=hidden_dims, depth=depth).to(device)

    model_clfr = classifier_model(input_dims=output_dims, no_classes=no_classes).to(device)

    if args.load:
        checkpoint = torch.load(args.file_path_save_src_feats)
        model_feats.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(args.file_path_save_src_clfr)
        model_clfr.load_state_dict(checkpoint['model_state_dict'])

    elif args.train:
        if args.train_pre_acq_model:
            if q == 0:
                'Only run the inital training for the first run if running through all models'
                for k in range(0,args.no_runs):
                    #new model initializations
                    model_feats = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                                            hidden_dims=hidden_dims, depth=depth).to(device)

                    model_clfr = classifier_model(input_dims=output_dims, no_classes=no_classes).to(device)

                    _, _,pool = train_active_loop(model_feats=model_feats, device=device, args=args,
                                                                model_clfr=model_clfr,
                                                                train_dataloader=train_loader,
                                                                wanb_con=wandb, val_dataload=val_loader,total_budget=args.no_queries,
                                                                     queries_per_round=args.no_queries)
                    save_path_init_model_feats = os.path.join(args.path_pre_active_model,f'save_model_seed_feats_{k}_{args.dataset}')
                    save_path_init_model_clfr = os.path.join(args.path_pre_active_model, f'save_model_seed_clfr_{k}_{args.dataset}')
                    save_path_init_pool =  os.path.join(args.path_pre_active_model, f'Initial_label_pool_seed_{k}_{args.dataset}')

                    torch.save({
                        'model_state_dict': model_feats.state_dict(),
                    }, save_path_init_model_feats)
                    torch.save({
                        'model_state_dict': model_clfr.state_dict(),
                    }, save_path_init_model_clfr)

                    np.save(save_path_init_pool,pool)

                    print(f"Saving Initial Model and pool for run number: {k}")
        for k in range(0,args.no_runs):
            #wandb.init(config=args_dict)

            #adding run number into the args to pass into the main loop
            args.run_no = k
            print(f"Run number: {k} for {dict_type[args.query_type]}")
            load_path_init_model_feats = os.path.join(args.path_pre_active_model, f'save_model_seed_feats_{k}_{args.dataset}')
            load_path_init_model_clfr = os.path.join(args.path_pre_active_model, f'save_model_seed_clfr_{k}_{args.dataset}')
            load_path_init_pool = os.path.join(args.path_pre_active_model, f'Initial_label_pool_seed_{k}_{args.dataset}')

            #Load pre trained models on initial pool for all methods
            checkpoint = torch.load(load_path_init_model_feats)
            model_feats.load_state_dict(checkpoint['model_state_dict'])
            checkpoint = torch.load(load_path_init_model_clfr)
            model_clfr.load_state_dict(checkpoint['model_state_dict'])

            #load label pool for trianing initial model
            pool = np.load(load_path_init_pool+'.npy')


            f1_score_list,acc_list,_ = train_active_loop(model_feats=model_feats, device=device, args=args, model_clfr=model_clfr,
                                      train_dataloader=train_loader,
                                     wanb_con=wandb, val_dataload=val_loader,init_pool=pool,val_each_epoch=False)


            f1_list_runs.append(f1_score_list)
            acc_list_runs.append(acc_list)

    np.savez(full_path_save_results, acc_list = np.asarray(acc_list_runs) , f1_list = np.asarray(f1_score_list),total_budget =args.total_budget,
             no_queries_round = args.no_queries)





