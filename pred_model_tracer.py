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
parser.add_argument('--epoch', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nhid', default=100, type=int, metavar='nh',
                    help='hidden layer size in the network cell')
parser.add_argument('--nlag', default=1, type=int, metavar='nL',
                    help='lag steps involved in the prediction of next state')
parser.add_argument('--nloss', default=50, type=int, metavar='nv',
                    help='number of steps to measure in the loss function')
parser.add_argument('--nstep', default=1, type=int, metavar='Np',
                    help='forward time steps in model prediction')
parser.add_argument('--npred', default=1, type=int, metavar='Np',
                    help='forward in the prediction interval (fixed as 1)')
# optimization parameters
parser.add_argument('--loss-type', '--lt', default='mixed', type=str, metavar='LT',
                    help='choices of loss functions (kld, mse, mixed, obsd)')

# checkpoints/data setting
parser.add_argument('-c', '--checkpoint', default='checkpoint/BARO2_h10sk0sU1', type=str, metavar='C_PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--data-file', default='data/baro2_h10sk0sU1_dk1dU1', type=str, metavar='DATA_PATH',
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


fname = 'flow02_ls4stg0hs{}lag{}st{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag, args.nstep,  args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
fnamet = 'tracer02_ls4stg0hs{}lag{}st{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag, args.nstep,  args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
fnameo = 'sU1_ls4stg0hs{}lag{}st{}nl{}_seq{}nsk{}np{}_epoch{}_'.format(args.nhid, args.nlag, args.nstep,  args.nloss,
            args.input_length, args.nskip, args.npred, args.epoch) + args.loss_type
def main():
    # model
    model1 = models.LSTMupdate(input_size = 3, hidden_size = args.nhid, output_size = 2, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    model2 = models.LSTMupdate(input_size = 3, hidden_size = args.nhid, output_size = 2,
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    # models for tracers
    tracer1 = models.LSTMupdate(input_size = 5, hidden_size = args.nhid, output_size = 2, 
                              nlags = args.nlag, nlayers = 4, nstages = 0).double()
    tracer2 = models.LSTMupdate(input_size = 5, hidden_size = args.nhid, output_size = 2, 
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
    tracer1.to(dev1)
    tracer2.to(dev2)
    
    # load the pretrained model
    model_path1 = torch.load(os.path.join(cfg['checkpoint'], 'model1_'+fname), map_location=dev1)
    model1.load_state_dict(model_path1['model_state_dict'])
    model_path2 = torch.load(os.path.join(cfg['checkpoint'], 'model2_'+fname), map_location=dev2)
    model2.load_state_dict(model_path2['model_state_dict'])
    tracer_path1 = torch.load(os.path.join(cfg['checkpoint'], 'model1_'+fnamet), map_location=dev1)
    tracer1.load_state_dict(tracer_path1['model_state_dict'])
    tracer_path2 = torch.load(os.path.join(cfg['checkpoint'], 'model2_'+fnamet), map_location=dev2)
    tracer2.load_state_dict(tracer_path2['model_state_dict'])
    model = (model1, model2)
    tracer = (tracer1, tracer2)

    # load dataset
    data_load = scipy.io.loadmat(args.data_file)
    tt = np.transpose(data_load.get('tout'), (1,0))
    samp_u = np.transpose(data_load.get('Uout'), (1,0))
    samp_w = np.transpose(data_load.get('wout'), (1,0))
    samp_t = np.transpose(data_load.get('Trout'), (1,0))
    noise  = np.transpose(data_load.get('noise'), (1,0))
    #tt     =     tt[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    #samp_u = samp_u[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    #samp_w = samp_w[args.train_length*args.nskip:(args.seq_length_pred+args.train_length)*args.nskip:args.nskip]
    #noise  = np.sum(noise[1+args.train_length*args.nskip:1+(args.seq_length_pred+args.train_length)*args.nskip].reshape(-1,args.nskip), axis=1)
    tt     =     tt[:(args.seq_length_pred)*args.nskip:args.nskip]
    samp_u = samp_u[:(args.seq_length_pred)*args.nskip:args.nskip]
    samp_w = samp_w[:(args.seq_length_pred)*args.nskip:args.nskip]
    samp_t = samp_t[:(args.seq_length_pred)*args.nskip:args.nskip]
    noise  = np.sum(noise[1:1+(args.seq_length_pred)*args.nskip].reshape(-1,args.nskip), axis=1)
    tt = tt - tt[0]
        
    traj_u = torch.from_numpy(samp_u)
    traj_w = torch.from_numpy(samp_w)
    traj_t = torch.from_numpy(samp_t)
    traj_n = torch.from_numpy(noise)[:,None]
    traj_set = torch.cat([traj_u,traj_w,traj_t,traj_n], 1)
    print(traj_set.size())
    init1 = torch.zeros(args.input_length, 1, 3, dtype=torch.double)
    init2 = torch.zeros(args.input_length, 1, 3, dtype=torch.double)
    initU = torch.zeros(args.input_length, 1, 5, dtype=torch.double)
    tinit1 = torch.zeros(args.input_length, 1, 5, dtype=torch.double)
    tinit2 = torch.zeros(args.input_length, 1, 5, dtype=torch.double)
    init1[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    init2[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    initU[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    init1[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, :2])
    init2[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, 2:])
    initU[:, 0, 1:] = torch.from_numpy(samp_w[:args.input_length, :])
    
    tinit1[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    tinit2[:, 0, 0]  = torch.from_numpy(samp_u[:args.input_length,0])
    tinit1[:, 0, 1:3] = torch.from_numpy(samp_w[:args.input_length, :2])
    tinit2[:, 0, 1:3] = torch.from_numpy(samp_w[:args.input_length, 2:])
    tinit1[:, 0, 3:] = torch.from_numpy(samp_t[:args.input_length, :2])
    tinit2[:, 0, 3:] = torch.from_numpy(samp_t[:args.input_length, 2:])
    init_set = (init1, init2, initU, tinit1,tinit2)    

    # inference
    traj_pred, err_pred = prediction(init_set, traj_set, model,tracer, device)
    err_ave = np.zeros(err_pred.shape)
    for i in range(err_pred.shape[0]):
        err_ave[i] = np.mean(err_pred[max(0,i-10):i], axis=0)
        
    datapath = os.path.join(cfg['checkpoint'], 'trajU_sU1_' + fnameo)
    np.savez(datapath, tt = tt, traj_pred = traj_pred, err_pre = err_pred, err_ave = err_ave,
             samp_u = samp_u, samp_w = samp_w, samp_t = samp_t) 
    
    
def prediction(init_set, targ_set, model,tracer, device):
    model1, model2 = model
    tracer1, tracer2 = tracer
    dev1, dev2 = device
    with torch.no_grad():
        model1.eval()
        model2.eval()
        tracer1.eval()
        tracer2.eval()
    baroU = BaroTopo(dt = args.nskip*0.01, H0=10, device=dev1)

    istate1, istate2, inputs, tstate1, tstate2 = init_set
    
    traj_pred = np.zeros((args.seq_length_pred, 9))
    errors    = np.zeros((args.seq_length_pred, 5))
    traj_pred[:args.input_length,:5]  = inputs.data.cpu().numpy()[:args.input_length,0]
    traj_pred[:args.input_length,5:7] = tstate1.data.cpu().numpy()[:args.input_length,0,-2:]
    traj_pred[:args.input_length,7:]  = tstate2.data.cpu().numpy()[:args.input_length,0,-2:]
    istate1, istate2, inputs = istate1.to(dev1), istate2.to(dev2), inputs.to(dev1)
    tstate1, tstate2 = tstate1.to(dev1), tstate2.to(dev2)
    target = targ_set
    hidden1, hidden2 = (), ()
    thidden1, thidden2 = (), ()
    with torch.no_grad():
        for start in range(args.input_length, args.seq_length_pred):
            # run model in one forward iteration
            noise  = target[start-args.input_length: start, 9].unsqueeze(1).to(dev1)
            # hidden1, hidden2 = (), ()
            for im in range(args.nstep):
                output1, hidden1 = model1(istate1, hidden1, device=dev1)
                output2, hidden2 = model2(istate2, hidden2, device=dev2)
                inputs[:,:,1:3] = (inputs[:,:,1:3]+output1) / 2
                inputs[:,:,3:]  = (inputs[:,:,3:] +output2.to(dev1)) / 2
                outputU = inputs[:,:,0] + baroU.baro_euler(inputs, noise)
                # update tracer
                toutput1, thidden1 = tracer1(tstate1, thidden1, device=dev1)
                toutput2, thidden2 = tracer2(tstate2, thidden2, device=dev2)
                
            out2, tout2 = output2.to(dev1), toutput2.to(dev1)
            output = torch.cat([output1, out2], 2)
            toutput = torch.cat([toutput1, tout2], 2) 
    
            pred = output.data.cpu().numpy()[-1,0,:]
            tpred = toutput.data.cpu().numpy()[-1,0,:]
            predU = outputU.data.cpu().numpy()[-1,0]
            targ = target.data.cpu().numpy()[start,:9]
            traj_pred[start,1:5] = pred
            traj_pred[start,5:] = tpred
            traj_pred[start,0] = predU
            errors[start,0] = np.square(predU - targ[0])
            errors[start,1] = np.square(pred[:2]-targ[1:3]).sum()
            errors[start,2] = np.square(pred[2:]-targ[3:5]).sum()
            errors[start,3] = np.square(tpred[:2]-targ[5:7]).sum()
            errors[start,4] = np.square(tpred[2:]-targ[7:]).sum()
            
            istate1[:,0,0]  = target[start-args.input_length+1:start+1,0]
            istate2[:,0,0]  = target[start-args.input_length+1:start+1,0]
            tstate1[:,0,0]  = target[start-args.input_length+1:start+1,0]
            tstate2[:,0,0]  = target[start-args.input_length+1:start+1,0]
            inputs[:,0,0]   = target[start-args.input_length+1:start+1,0]
            # istate1[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            # istate2[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            # tstate1[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            # tstate2[:,0,0]  = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            # inputs[:,0,0]   = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,0])
            istate1[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:3])
            istate2[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,3:5])
            inputs[:,0,1:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:5])
            tstate1[:,0,1:3] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,1:3])
            tstate2[:,0,1:3] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,3:5])
            tstate1[:,0,3:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,5:7])
            tstate2[:,0,3:] = torch.from_numpy(traj_pred[start-args.input_length+1:start+1,7:])
    
            if start % 10 == 0:
                print('step {}: error_s = {:.6f},error_t = {:.6f}, error_U = {:.6f}'.format(start, 
                np.square(pred - targ[1:5]).mean(),np.square(tpred - targ[5:]).mean(), np.square(predU - targ[0])))

    return traj_pred, errors

if __name__ == '__main__':
    main()
