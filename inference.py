"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os

# Adjustment 2: Add tqdm for progress tracking

if os.getenv("USE_TQDM_NOTEBOOK") == "1":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# Adjustment 6: Wrap into a function to run in the Jupyter Notebook 
def infer_model(override_args):
    default_args = argparse.Namespace(
        dataset='yelp_clean',
        data_path='./datasets/',
        batch_size=400,
        topN='[10, 20, 50, 100]',
        tst_w_val=False, 
        cuda=False,        
        gpu='0',
        log_name='log',

        # params for diffusion
        mean_type='x0',
        steps=100,
        noise_schedule='linear-var',
        noise_scale=0.1,
        noise_min=0.0001,
        noise_max=0.02,
        sampling_noise=False,
        sampling_steps=0,

        # model
        model_path='./checkpoints/',
        model_name='yelp_clean.pth'
    )

    merged_args = {**vars(default_args), **vars(override_args)}
    args = argparse.Namespace(**merged_args)
    print("args:", args)
    
    args.data_path = args.data_path + args.dataset + '/'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###
    train_path = args.data_path + 'train_list.npy'
    valid_path = args.data_path + 'valid_list.npy'
    test_path = args.data_path + 'test_list.npy'

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)

    # Adjustment 4: Replace .A (deprecated) with .array(), and disable multiprocessing in train_loader
    np_data = torch.from_numpy(train_data.toarray())
    train_dataset = data_utils.DataDiffusion(np_data)

    #Adjustment 9: Disable multiprocessing in train_loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    if args.tst_w_val:
        # Adjustment 4: Replace .A (deprecated) with .array()
        np_tv_data = np_data + torch.from_numpy(valid_y_data.toarray())
        tv_dataset = data_utils.DataDiffusion(np_tv_data)

        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True) # Add pin_memory=True
    mask_tv = train_data + valid_y_data

    print('data ready.')


    ### CREATE DIFFUISON ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, device)
    diffusion.to(device)

    ### CREATE DNN ###

    # Adjustment 11: Change the directory of checkpoints
    # model_path = "../checkpoints/DiffRec/"
    model_path = args.model_path
    model_name = args.model_name

    # Adjustment 10: Inference requires the explicit model name
    # if args.dataset == "amazon-book_clean":
    #     model_name = "amazon-book_clean_lr5e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.0001_min0.0005_max0.005_sample0_reweight0_log.pth"  #The filename here contains a minor error. The actual hyperparameter 'reweight=1' is used during training.
    # elif args.dataset == "yelp_clean":
    #     model_name = "yelp_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.01_min0.001_max0.01_sample0_reweight0_log.pth"  # The filename here contains a minor error. The actual hyperparameter 'reweight=1' is used during training.
    # elif args.dataset == "ml-1m_clean":
    #     model_name = "ml-1m_clean_lr0.001_wd0.0_bs400_dims[200,600]_emb10_x0_steps40_scale0.005_min0.005_max0.01_sample0_reweight1_log.pth"
    # elif args.dataset == "amazon-book_noisy":
    #     model_name = "amazon-book_noisy_lr5e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps10_scale0.005_min0.0001_max0.0005_sample0_reweight1_log.pth"
    # elif args.dataset == "yelp_noisy":
    #     model_name = "yelp_noisy_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.001_min0.0005_max0.01_sample0_reweight0_log.pth"
    # elif args.dataset == "ml-1m_noisy":
    #     model_name = "ml-1m_noisy_lr0.001_wd0.0_bs400_dims[200,600]_emb10_x0_steps5_scale0.5_min0.001_max0.01_sample0_reweight0_log.pth"


    # Adjustment 12: disable weights_only
    model = torch.load(model_path + model_name, weights_only=False).to(device)

    print("models ready.")

    def evaluate(data_loader, data_te, mask_his, topN):
        model.eval()
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predict_items = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference"):
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]

                # Adjustment 5: Convert to float32 on GPU, use non_blocking
                batch = batch.to(device, dtype=torch.float32, non_blocking=True)

                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                prediction[his_data.nonzero()] = -np.inf

                _, indices = torch.topk(prediction, topN[-1])
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

        test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

        return test_results

    valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
    if args.tst_w_val:
        test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
    else:
        test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
    evaluate_utils.print_results(None, valid_results, test_results)
    
    return valid_results, test_results

# Adjustment 14: Add if __name__ == "__main__" to prevent recursively import
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')

    # Adjustment 10: Inference requires the explicit model name
    parser.add_argument ('--model_name', type=str, default='yelp_clean.pth', help='the .pth filename containing model state')

    # Adjustment 13: Inference requires explicit steps and noise configs

    # if args.dataset == 'amazon-book_clean':
    #     args.steps = 5
    #     args.noise_scale = 0.0001
    #     args.noise_min = 0.0005
    #     args.noise_max = 0.005
    # elif args.dataset == 'yelp_clean':
    #     args.steps = 5
    #     args.noise_scale = 0.01
    #     args.noise_min = 0.001
    #     args.noise_max = 0.01
    # elif args.dataset == 'ml-1m_clean':
    #     args.steps = 40
    #     args.noise_scale = 0.005
    #     args.noise_min = 0.005
    #     args.noise_max = 0.01
    # elif args.dataset == 'amazon-book_noisy':
    #     args.steps = 10
    #     args.noise_scale = 0.005
    #     args.noise_min = 0.0001
    #     args.noise_max = 0.0005
    # elif args.dataset == 'yelp_noisy':
    #     args.steps = 5
    #     args.noise_scale = 0.001
    #     args.noise_min = 0.0005
    #     args.noise_max = 0.01
    # elif args.dataset == 'ml-1m_noisy':
    #     args.steps = 5
    #     args.noise_scale = 0.5
    #     args.noise_min = 0.001
    #     args.noise_max = 0.01
    # if args.dataset == 'gowalla':
    #     args.steps = 5
    #     args.noise_scale = 0.0001
    #     args.noise_min = 0.0005
    #     args.noise_max = 0.005
    # if args.dataset == 'tmall':
    #     args.steps = 5
    #     args.noise_scale = 0.0001
    #     args.noise_min = 0.0005
    #     args.noise_max = 0.005
    # else:
    #     raise ValueError

    
    args = parser.parse_args()

    infer_model(args)



