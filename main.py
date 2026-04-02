"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

# Adjustment 2: Add tqdm for progress tracking

if os.getenv("USE_TQDM_NOTEBOOK") == "1":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
def train_model(override_args):
    default_args = argparse.Namespace(
        dataset='yelp_clean',
        data_path='./datasets/',
        lr=0.0001,
        weight_decay=0.0,
        batch_size=400,
        infer_batch_size=0,
        epochs=1000,
        topN='[10, 20, 50, 100]',
        tst_w_val=False,
        cuda=False,
        gpu='0',
        save_path='./saved_models/',
        log_name='log',
        round=1,
        loss_logging=True, # Adjustment 17: Add argument for loss and cost per epoch control

        # params for the model
        time_type='cat',
        dims='[1000]',
        norm=False,
        emb_size=10,

        # params for diffusion
        mean_type='x0',
        steps=100,
        noise_schedule='linear-var',
        noise_scale=0.1,
        noise_min=0.0001,
        noise_max=0.02,
        sampling_noise=False,
        sampling_steps=0,
        reweight=True
    )

    merged_args = {**vars(default_args), **vars(override_args)}
    args = argparse.Namespace(**merged_args)

    print("args:", args)

    #Adjustment 16: separate test batch_size with train batch_size, increasing inference throughput
    infer_batch_size = args.batch_size if args.infer_batch_size == 0 else args.infer_batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    def evaluate(data_loader, data_te, mask_his, topN):
        model.eval()
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predict_items = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", leave=False):
                # his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                
                #Adjustment 16: separate test batch_size with train batch_size, increasing inference throughput
                his_data = mask_his[e_idxlist[batch_idx*data_loader.batch_size:batch_idx*data_loader.batch_size+len(batch)]]


                # Adjustment 5: Convert to float32 on GPU, use non_blocking
                batch = batch.to(device, dtype=torch.float32, non_blocking=True)


                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                prediction[his_data.nonzero()] = -np.inf

                # Adjustment 7: Use the first result in the Recall@N array as selection 
                _, indices = torch.topk(prediction, topN[0])
                
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

        test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

        return test_results


    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###
    train_path = args.data_path + args.dataset + '/' + 'train_list.npy'
    valid_path = args.data_path + args.dataset + '/' + 'valid_list.npy'
    test_path = args.data_path + args.dataset + '/' + 'test_list.npy'

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)

    # Adjustment 4: Replace .A (deprecated) with .array()
    np_data = torch.from_numpy(train_data.toarray())

    train_dataset = data_utils.DataDiffusion(np_data)

    #Adjustment 9: Disable multiprocessing in train_loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(train_dataset, batch_size=infer_batch_size, shuffle=False, pin_memory=True) # Add pin_memory=True
    #Adjustment 16: separate test batch_size with train batch_size, increasing inference throughput
    
    if args.tst_w_val:
        # Adjustment 4: Replace .A (deprecated) with .array()
        np_tv_data = np_data + torch.from_numpy(valid_y_data.toarray())
        tv_dataset = data_utils.DataDiffusion(np_tv_data)

        #Adjustment 16: separate test batch_size with train batch_size, increasing inference throughput
        test_twv_loader = DataLoader(tv_dataset, batch_size=infer_batch_size, shuffle=False, pin_memory=True) # Add pin_memory=True
    mask_tv = train_data + valid_y_data

    print('data ready.')


    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

    ### Build MLP ###
    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("models ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)



    best_recall, best_epoch = -100, 0
    best_test_result = None
    print("Start training...")
    print('==='*18)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        if epoch - best_epoch >= 20: 
        # if epoch - best_epoch >= 20:
            tqdm.write('-'*18)
            tqdm.write('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches", leave=False):

            # Adjustment 5: Convert to float32 on GPU, use non_blocking
            batch = batch.to(device, dtype=torch.float32, non_blocking=True)


            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        # Adjustment 8: Move the evaluation results to behind the epoch results
        if args.loss_logging:
            tqdm.write("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                    "%H: %M: %S", time.gmtime(time.time()-start_time)))
            
        if epoch % 5 == 0:
            if not args.loss_logging:
                tqdm.write(f"Epoch: {epoch}")
                
            valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
            if args.tst_w_val:
                test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
            else:
                test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))

            evaluate_utils.print_results(None, valid_results, test_results)

            # Adjustment 7: Use the first result in the Recall@N array as selection 
            if valid_results[1][0] > best_recall: # recall@20 as selection
                best_recall, best_epoch = valid_results[1][0], epoch
                best_results = valid_results
                best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, '{}{}_lr{}_wd{}_bs{}_ibs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                    .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, infer_batch_size, args.dims, args.emb_size, args.mean_type, \
                    args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))

        if args.loss_logging or epoch % 5 == 0:
            tqdm.write('---'*18)

    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # Adjustment 15: Inform the filename
    print("Best model saved to ", '{}{}_lr{}_wd{}_bs{}_ibs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_{}.pth' \
                    .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, infer_batch_size, args.dims, args.emb_size, args.mean_type, \
                    args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.log_name))

# Adjustment 14: Add if __name__ == "__main__" to prevent recursively import
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400),
    parser.add_argument('--infer_batch_size', type=int, default=0, help='batch size when inference (equal to batch_size by default)')
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')
    parser.add_argument('--round', type=int, default=1, help='record the experiment')

    # params for the model
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    args = parser.parse_args()
    
    train_model(args)



