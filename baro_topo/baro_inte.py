"""
Forward Euler integrator for the topographic barotropic model
"""
import numpy as np
import torch

__all__ = ['BaroTopo']

class BaroTopo(object):
    "compute the one-step forward integration for the topographic barotropic model"
    def __init__(self, dt, H0 = 10, damp = 0.0125, device='cpu'):
        self.dt = dt
        self.hk = torch.tensor([H0*1/2, H0*.5/2], dtype=torch.double, device=device)
        self.damp = damp
        self.kvec = np.array([1, 2])
        
    def baro_euler(self, inputs, noise):
        nseq  = inputs.shape[0]
        nsamp = inputs.shape[1]
        coeff = (self.hk/self.kvec).repeat(nseq, nsamp, 1)
        omek = torch.stack([inputs[:,:,2]+inputs[:,:,1], inputs[:,:,4]+inputs[:,:,3]], 2)
        # forcing = noise
        
        FU = 2 * (omek * coeff).sum(axis=2) - self.damp * inputs[:,:,0]
        inte = FU * self.dt + noise
        return inte