from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch.utils import data

import math
import numpy as np
import random
import scipy.io

import argparse
import os

from utils import Logger, AverageMeter, accuracy, mkdir_p
import models
from baro_topo import BaroTopo

from collections import OrderedDict

# hyper parameters for prediction
parser = argparse.ArgumentParser(description='model prediction')
parser.add_argument('--seq_length_pred', default=20000, type=int, metavar='T',
                    help='length of prediction sequence')
parser.add_argument('--train_length', default=10100, type=int, metavar='T',
                    help='sequence length for training samples')
# data loading parameters
parser.add_argument('--input_length', default=100, type=int, metavar='L',
                    help='model input length')
parser.add_argument('--nskip', default=10, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
# (train-batch * iters = train_length - input_length)
# model parameters
parser.add_argument('--epoch', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=100, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nlag', default=5, type=int, metavar='nL',
                    help='lag steps involved in the prediction of next state')
parser.add_argument('--nloss', default=100, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
parser.add_argument('--nstep', default=1, type=int, metavar='Np',
                    help='forward time steps in model prediction')
parser.add_argument('--npred', default=1, type=int, metavar='Np',
                    help='forward in the prediction interval (fixed as 1)')
# optimization parameters
parser.add_argument('--loss-type', '--lt', default='mixed', type=str, metavar='LT',
                    help='choices of loss functions (kld, mse, mixed, obsd)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/ModelU_BARO2sk0sU10', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro2_sk0sU10_dk1dU1', type=str, metavar='DATA_PATH',
                    help='path to data set (default: baro2)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
cfg = {k: v for k, v in args._get_kwargs()}

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

fname = 'full_ls4stg0hs{}lag{}st{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag, args.nstep,  args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
def main():
    # model
    model1 = models.LSTMupdate(input_size = 3, hidden_size = args.nhid, output_size = 2, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    model2 = models.LSTMupdate(input_size = 3, hidden_size = args.nhid, output_size = 2,
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    # load model on GPU
    dev1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev2 = dev1
    #dev2 = torch.device("cuda:1" if torch.cuda.device_count()>1 else "cuda:0")
    device = (dev1, dev2)
    print('This code is run by {} and {}'.format(dev1, dev2))
    # if  torch.cuda.device_count() > 1:
    #     print('using ', torch.cuda.device_count(), ' GPUs!')
    #     model = nn.DataParallel(model).to(device)
    #     cudnn.benchmark = True
    model1.to(dev1)
    model2.to(dev2)
    
    # load the pretrained model
    model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'model1_'+fname), map_location=dev1)
    model1.load_state_dict(model_path1['model_state_dict'])
    model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'model2_'+fname), map_location=dev2)
    model2.load_state_dict(model_path2['model_state_dict'])
    model = (model1, model2)

    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    tt = np.transpose(data_load.get('tout'), (1,0))
    samp_u = np.transpose(data_load.get('Uout'), (1,0))
    samp_w = np.transpose(data_load.get('wout'), (1,0))
    noise  = np.transpose(data_load.get('noise'), (1,0))
    tt     =     tt[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    samp_u = samp_u[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    samp_w = samp_w[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    noise  = np.sum(noise[1+args.train_length*args.nskip:1+(args.seq_length_pred+args.train_length)*args.nskip].reshape(-1,args.nskip), axis=1)
    tt = tt - tt[0]
        
    traj_u = torch.from_numpy(samp_u)
    traj_w = torch.from_numpy(samp_w)
    traj_n = torch.from_numpy(noise)[:,None]
    traj_set = torch.cat([traj_u,traj_w,traj_n], 1)
    print(traj_set.size())
    init1 = torch.zeros(args.input_length, 1, 3, dtype=torch.double)
    init2 = torch.zeros(args.input_length, 1, 3, dtype=torch.double)
    initU = torch.zeros(args.input_length, 1, 5, dtype=torch.double)
    init1[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    init2[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    initU[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    init1[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, :2])
    init2[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, 2:])
    initU[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, :])
    init_set = (init1, init2, initU)    

    # inference
    traj_pred, err_pred = prediction(init_set, traj_set, model, device)
    err_ave = np.zeros(err_pred.shape)
    for i in range(err_pred.shape[0]):
        err_ave[i] = np.mean(err_pred[max(0,i-10):i], axis=0)
        
    datapath = os.path.join(cfg['checkpoint'], 'traj_' + fname)
    np.savez(datapath, tt = tt, traj_pred = traj_pred, err_pre = err_pred, err_ave = err_ave,
             samp_u = samp_u, samp_w = samp_w) 
    
    
def prediction(init_set, targ_set, model, device):
    model1, model2 = model
    dev1, dev2 = device
    with torch.no_grad():
        model1.eval()
        model2.eval()
    baroU = BaroTopo(dt = args.nskip*0.01, device=dev1)

    istate1, istate2, inputs = init_set
    
    traj_pred = np.zeros((args.seq_length_pred, 5))
    errors    = np.zeros((args.seq_length_pred, 3))
    traj_pred[:args.input_length] = inputs.data.cpu().numpy()[:args.input_length,0]
    istate1, istate2, inputs = istate1.to(dev1), istate2.to(dev2), inputs.to(dev1)
    target = targ_set
    hidden1, hidden2 = (), ()
    with torch.no_grad():
        for start in range(args.input_length, args.seq_length_pred):
            # run model in one forward iteration
            noise  = target[start-args.input_length: start, 5].to(dev1)
            # hidden1, hidden2 = (), ()
            for im in range(args.nstep):
                output1, hidden1 = model1(istate1, hidden1, device=dev1)
                output2, hidden2 = model2(istate2, hidden2, device=dev2)
                inputs[:,:,1:3] = (inputs[:,:,1:3]+output1) / 2
                inputs[:,:,3:]  = (inputs[:,:,3:] +output2.to(dev1)) / 2
                outputU = inputs[:,:,0] + baroU.baro_euler(inputs, noise)
                
            out2 = output2.to(dev1)
            output = torch.cat([output1, out2], 2)
    
            pred = output.data.cpu().numpy()[-1,0,:]
            predU = outputU.data.cpu().numpy()[-1,0]
            targ = target.data.cpu().numpy()[start,:5]
            traj_pred[start,1:] = pred
            traj_pred[start,0] = predU
            errors[start,0] = np.square(predU - targ[0]).sum()
            errors[start,1] = np.square(pred[:2]-targ[1:3]).sum()
            errors[start,2] = np.square(pred[2:]-targ[3:]).sum()
            
            # istate1[:,0,0]  = target[start-args.input_length+1:start+1,0]
            # istate2[:,0,0]  = target[start-args.input_length+1:start+1,0]
            # inputs[:,0,0]  = target[start-args.input_length+1:start+1,0]
            istate1[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            istate2[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            inputs[:,0,0]   = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            istate1[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:3])
            istate2[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,3:])
            inputs[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:])
    
            if start % 100 == 0:
                print('step {}: error = {:.6f}'.format(start, np.square(pred - targ[1:]).mean()))

    return traj_pred, errors

if __name__ == '__main__':
    main()
