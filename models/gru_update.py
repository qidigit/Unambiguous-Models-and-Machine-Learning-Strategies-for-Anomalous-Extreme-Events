import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

__all__ = ['LSTMupdate', 'LSTMnoise']

def add_cell_block(input_size, hidden_size, expan = 1):
    lin = nn.Linear(input_size, expan * hidden_size)
    relu = nn.ReLU(inplace=True)
    
    return [lin, relu]

class inner_cell(nn.Module):
    def __init__(self, input_size, hidden_size, nstages = 1, out_ch = 4):
        """"Inner structure in the LSTM Net"""
        super(inner_cell, self).__init__()
        seq_list = []
        for i in range(nstages):
            if i==0:
                seq_list = add_cell_block(input_size, hidden_size, 1)
            else:
                seq_list += add_cell_block(1 * hidden_size, hidden_size, 1)
           
        if nstages > 0:
            seq_list.append(nn.Linear(1 * hidden_size, out_ch * hidden_size))
        else:
            seq_list.append(nn.Linear(input_size, out_ch * hidden_size))
        self.seq = nn.Sequential(*seq_list)
        
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers):
        """"LSTM Net cell with multiple layers"""
        super(LSTMCell, self).__init__()

        self.hsize = hidden_size
        self.nlayers = nlayers
        # hw = torch.empty(nlayers, dtype=torch.double)
        # nn.init.ones_(hw)
        # cw = torch.empty(nlayers, dtype=torch.double)
        # nn.init.ones_(cw)
        # self.hlw = nn.Parameter(hw)
        # self.clw = nn.Parameter(cw)
        # ihw = torch.empty(nlayers, nlayers, dtype=torch.double)
        # nn.init.eye_(ihw)
        # icw = torch.empty(nlayers, nlayers, dtype=torch.double)
        # nn.init.eye_(icw)
        # self.ihlw = nn.Parameter(ihw)
        # self.iclw = nn.Parameter(icw)
        
        ih, hh, ch = [], [], []
        hlink, clink = [], []
        for i in range(nlayers):
            ih.append(inner_cell(input_size, hidden_size, 0, 4))
            if i == 0:
                hh.append(inner_cell(hidden_size, hidden_size, 0, 4))
                ch.append(inner_cell(hidden_size, hidden_size, 0, 3))
            else:
                hh.append(inner_cell(i * hidden_size, hidden_size, 0, 4))
                ch.append(inner_cell(i * hidden_size, hidden_size, 0, 3))
            hlink.append(nn.Linear(hidden_size, hidden_size))
            clink.append(nn.Linear(hidden_size, hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)
        self.w_ch = nn.ModuleList(ch)
        self.hlw = nn.ModuleList(hlink)
        self.clw = nn.ModuleList(clink)

    def forward(self, inputs, hidden):
        """"Defines the forward computation of the LSTMCell"""     
        # hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0], hidden[1]
            cxi = cx[:, -self.hsize:]
            gates = self.w_ih[i](inputs) + self.w_hh[i](hx)
            gates1 = self.w_ch[i](cx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate1, f_gate1, o_gate1 = gates1.chunk(3, 1)
            i_gate = torch.sigmoid(i_gate + i_gate1)
            f_gate = torch.sigmoid(f_gate + f_gate1)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate + o_gate1)
            ncx = (f_gate * cxi) + (i_gate * c_gate)
            nhx = o_gate * torch.tanh(ncx)
            
            if i == 0:
                nhy, ncy = nhx, ncx
            else:
                nhy = torch.cat([nhy, nhx], 1)
                ncy = torch.cat([ncy, ncx], 1)
            # cy.append(ncx)
            # hy.append(nhx)
            # nhy = torch.mul(self.ihlw[i, i], nhx)
            # ncy = torch.mul(self.iclw[i, i], ncx)
            # for l in range(i):
            #     nhy += torch.mul(self.ihlw[i][l], hy[l])
            #     ncy += torch.mul(self.iclw[i][l], cy[l])
            hidden = [nhy, ncy]
            
            if i == 0:                
                hout = self.hlw[i](nhx)  #* nhx
                cout = self.clw[i](ncx)  #* ncx
            else:
                hout += self.hlw[i](nhx)  #* nhx
                cout += self.clw[i](ncx)  #* ncx

        return hout, cout

    
class LSTMupdate(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_size, nlayers=4):
        super(LSTMupdate, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.lseq = seq_size
        self.lstmcell = LSTMCell(input_size, hidden_size, nlayers)
        # add output linear combinations
        lin_list = []
        for i in range(seq_size):
            lin_list.append(nn.Sequential(nn.Linear(hidden_size, 10 * hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(10 * hidden_size, output_size)) )
        self.linear = nn.ModuleList(lin_list)
        
    def forward(self, inputs, hidden = (), npred = 0, device = "cpu"):
        output = []
        
        stage = []
        # hout = []
        for i in range(inputs.size(0)):            
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double).to(device)
                    ct = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double).to(device)
                else:
                    ht, ct = hidden
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                # hout.append(ht)
            else:
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                # hout.append(ht)
                
            if i < self.lseq:
                stage.append(ht)
            else:
                stage[:-1] = stage[1:]
                stage[-1] = ht
            if i >= self.lseq-1:
                pred = inputs[i,:,1:] + self.linear[-1](stage[-1])
                for l in range(self.lseq-1):
                    pred += self.linear[l](stage[l])
                output.append(pred)
                
            if i == npred:
                hidden = (ht, ct)
                
        # for i in range(inputs.size(0)):
        #     if i >= self.lseq-1:
        #         pred = inputs[i,:,1:] + self.linear[-1](hout[i])
        #         for l in range(self.lseq-1):
        #             pred += self.linear[l](hout[i-l-1])
        #         output.append(pred)
                
        return torch.stack(output, 0), hidden
    
class LSTMnoise(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, u_size, noi_len,unoi, nlayers=4):
        super(LSTMnoise, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.lseq = u_size
        self.lstmcell = LSTMCell(input_size, hidden_size, nlayers)
        # add output linear combinations
        lin_list = []
        for i in range(u_size):
            lin_list.append(nn.Sequential(nn.Linear(hidden_size, 10 * hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(10 * hidden_size, output_size)) )
        self.linear = nn.ModuleList(lin_list)
        
        # noise parameter in the observed variable
        self.nlen = noi_len
        noi = torch.empty(noi_len, dtype=torch.double)
        nn.init.normal_(noi, mean=0.0, std=float(np.sqrt(unoi)))
        self.noise = nn.Parameter(noi)
        
    def forward(self, inputs, idx, hidden = (), npred = 1, device = "cpu"):
        output, uout = [], []
        
        stage = []
        # hout = []
        for i in range(inputs.size(0)):
            istate = torch.zeros(inputs.size(1), inputs.size(2), dtype=torch.double).to(device)
            istate[:,1:] = inputs[i,:,1:]
            istate[:,0] = inputs[i, :, 0] - self.noise[(i+np.array(idx))]
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double).to(device)
                    ct = torch.zeros(inputs.size(1), self.hidden_size, dtype=torch.double).to(device)
                else:
                    ht, ct = hidden
                ht, ct = self.lstmcell(istate, (ht, ct))
                # hout.append(ht)
            else:
                ht, ct = self.lstmcell(istate, (ht, ct))
                # hout.append(ht)
                
            if i < self.lseq:
                stage.append(ht)
            else:
                stage[:-1] = stage[1:]
                stage[-1] = ht
            if i >= self.lseq-1:
                pred = istate + self.linear[-1](stage[-1])
                for l in range(self.lseq-1):
                    pred += self.linear[l](stage[l])
                uout.append(pred[:,0])
                pred[:, 0] = pred[:,0] + self.noise[(npred+i+np.array(idx))]
                output.append(pred)
                
            if i == npred-1:
                hidden = (ht, ct)
                
        # for i in range(inputs.size(0)):
        #     if i >= self.lseq-1:
        #         pred = inputs[i,:,1:] + self.linear[-1](hout[i])
        #         for l in range(self.lseq-1):
        #             pred += self.linear[l](hout[i-l-1])
        #         output.append(pred)
                
        return torch.stack(output, 0), torch.stack(uout,0), hidden

if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        seq = torch.randn(10, 100, 5, dtype=torch.double)
        lstm = LSTMupdate(seq.size(2), 50, 4, 2).double()
        output_seq, hidden = lstm(seq)
        loss = output_seq.sum()
        loss.backward()
    print(output_seq.size())