from __future__ import print_function

import time
import torch
import torch.nn as nn
from torch.utils import data

import math
import numpy as np
import random
import scipy.io
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


import argparse
import os

from utils import Logger, AverageMeter, accuracy, mkdir_p
import models

from collections import OrderedDict

# hyper parameters for prediction
parser = argparse.ArgumentParser(description='model prediction')
parser.add_argument('--seq_length_pred', default=5000, type=int, metavar='T',
                    help='length of prediction sequence')
parser.add_argument('--train_length', default=10200, type=int, metavar='T',
                    help='sequence length for training samples')
# data loading parameters
parser.add_argument('--input_length', default=200, type=int, metavar='L',
                    help='model input length')
parser.add_argument('--nskip', default=50, type=int, metavar='nk',
                    help='time step skip in the loaded raw data, dt=1e-2')
# (train-batch * iters = train_length - input_length)
# model parameters
parser.add_argument('--epoch', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=100, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nlag', default=5, type=int, metavar='nL',
                    help='lag steps involved in the prediction of next state')
parser.add_argument('--nstep', default=1, type=int, metavar='Np',
                    help='forward time steps in model prediction')
parser.add_argument('--npred', default=1, type=int, metavar='Np',
                    help='forward in the prediction interval (fixed as 1)')
# optimization parameters
parser.add_argument('--loss-type', '--lt', default='mixed', type=str, metavar='LT',
                    help='choices of loss functions (kld, mse, mixed, obsd)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='results_training/sk0sU10_dk1du1/', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro2noise_sk0sU10_dk1dU1', type=str, metavar='DATA_PATH',
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

fname = 'ls4stg0hs{}lag{}st{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag, args.nstep,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
def main():
    # model
    model = models.LSTMupdate(input_size = 5, hidden_size = args.nhid, output_size = 5, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    # load model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('This code is run by {}'.format(device))
    # if  torch.cuda.device_count() > 1:
    #     print('using ', torch.cuda.device_count(), ' GPUs!')
    #     model = nn.DataParallel(model).to(device)
    #     cudnn.benchmark = True
    model.to(device)
    
    # load the pretrained model
    model_path = torch.load(os.path.join(cfg['checkpoint'], 'checkpoint_'+fname), map_location=device)
    model.load_state_dict(model_path['model_state_dict'])
    
    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    tt = np.transpose(data_load.get('tout'), (1,0))
    samp_u = np.transpose(data_load.get('Uout'), (1,0))
    samp_w = np.transpose(data_load.get('wout'), (1,0))
    tt     =     tt[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    samp_u = samp_u[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip, 0]
    samp_w = samp_w[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip, :]
    tt = tt - tt[0]
        
    traj_u = torch.from_numpy(samp_u)[:, None]
    traj_w = torch.from_numpy(samp_w)
    traj_set = torch.cat([traj_u,traj_w], 1)
    init_set = torch.zeros(args.input_length, 1, 5, dtype=torch.double)
    init_set[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length])
    init_set[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length])
        

    # inference
    traj_pred, err_pred = prediction(init_set, traj_set, model, device)
    err_ave = np.zeros(err_pred.shape)
    for i in range(err_pred.shape[0]):
        err_ave[i] = np.mean(err_pred[max(0,i-10):i], axis=0)
        
    # datapath = os.path.join(cfg['checkpoint'], 'traj_' + fname)
    # np.savez(datapath, tt = tt, traj_pred = traj_pred, err_pre = err_pred, err_ave = err_ave,
    #          samp_u = samp_u, samp_w = samp_w) 
    
    # draw results
    fig = plt.figure(100)
    fig.add_subplot(311)
    plt.plot(tt, err_ave[:,0], linewidth=1)
    plt.ylabel('rmse in U')
    fig.add_subplot(312)
    plt.plot(tt, err_ave[:,1], linewidth=1)
    plt.ylabel('rmse in mode 1')
    fig.add_subplot(313)
    plt.plot(tt, err_ave[:,2], linewidth=1)
    plt.xlabel('time')
    plt.ylabel('rmse in mode 2')
    plt.suptitle('model prediction error in trajectory solution')
    
    fig1 = plt.figure()
    fig1.add_subplot(311)
    plt.plot(tt, samp_u, linewidth=1)
    plt.ylabel('U')
    fig1.add_subplot(312)
    plt.plot(tt, samp_w[:,0], linewidth=1)
    plt.ylabel('Re mode 1')
    fig1.add_subplot(313)
    plt.plot(tt, samp_w[:,2], linewidth=1)
    plt.ylabel('Re mode 2')
    plt.xlabel('time')
    plt.suptitle('target trajectory of the two-mode topographic model')
    fig2 = plt.figure()
    fig2.add_subplot(311)
    plt.plot(tt, traj_pred[:,0], linewidth=1)
    plt.ylabel('U')
    fig2.add_subplot(312)
    plt.plot(tt, traj_pred[:,1], linewidth=1)
    plt.ylabel('Re mode 1')
    fig2.add_subplot(313)
    plt.plot(tt, traj_pred[:,3], linewidth=1)
    plt.ylabel('Re mode 2')
    plt.xlabel('time')
    plt.suptitle('model prediction from the trained neural network model')
    
    fig3 = plt.figure()
    fig3.add_subplot(211)
    plt.plot(tt, samp_w[:,1], linewidth=1)
    plt.ylabel('Im mode 1')
    fig3.add_subplot(212)
    plt.plot(tt, samp_w[:,3], linewidth=1)
    plt.ylabel('Im mode 2')
    plt.xlabel('time')
    plt.suptitle('target trajectory of the two-mode topographic model')
    fig4 = plt.figure()
    fig4.add_subplot(211)
    plt.plot(tt, traj_pred[:,2], linewidth=1)
    plt.ylabel('Im mode 1')
    fig4.add_subplot(212)
    plt.plot(tt, traj_pred[:,4], linewidth=1)
    plt.ylabel('Im mode 2')
    plt.xlabel('time')
    plt.suptitle('model prediction from the trained model')
    
    
def prediction(init_set, target, model, device="cpu"):

    with torch.no_grad():
        model.eval()
        
    traj_pred = np.zeros((args.seq_length_pred, 5))
    errors    = np.zeros((args.seq_length_pred, 3))
    traj_pred[:args.input_length + args.npred-1] = init_set.data.cpu().numpy()[:args.input_length + args.npred-1,0]
    inputs = init_set.to(device)
    with torch.no_grad():
        for start in range(args.input_length + args.npred-1, args.seq_length_pred):
            # run model in one forward iteration
            hidden = ()
            for im in range(args.nstep):
                if im == 0:
                    outputs, hidden = model(inputs, hidden, device=device)
                else:
                    outputs, hidden = model(outputs, hidden, device=device)
                    
            pred = outputs.data.cpu().numpy()[-1,0,:]
            targ = target.data.cpu().numpy()[start]
            traj_pred[start,:] = pred
            errors[start,0] = np.square(pred[0]-targ[0])
            errors[start,1] = np.square(pred[1:3]-targ[1:3]).sum()
            errors[start,2] = np.square(pred[3:]-targ[3:]).sum()
            
            inputs[:,0,0]  = target[start-args.input_length+1:start+1,0]
            # inputs[:,0,1:] = target[start-args.input_length+1:start+1,1:]
            inputs[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:])
    
            if start % 100 == 0:
                print('step {}: error = {:.6f}'.format(start, np.square(pred - targ).mean()))

    return traj_pred, errors

if __name__ == '__main__':
    main()