import torch.nn as nn
import torch

__all__ = ['LSTMdense']

def add_cell_block(input_size, hidden_size, expan = 1):
    lin = nn.Linear(input_size, expan * hidden_size)
    relu = nn.ReLU(inplace=True)
    
    return [lin, relu]

class inner_cell(nn.Module):
    def __init__(self, input_size, hidden_size, nstages = 1, out_ch = 4):
        """"Constructor of the class"""
        super(inner_cell, self).__init__()
        for i in range(nstages):
            if i==0:
                seq_list = add_cell_block(input_size, hidden_size, 1)
            else:
                seq_list += add_cell_block(1 * hidden_size, hidden_size, 1)
            
        seq_list.append(nn.Linear(1 * hidden_size, out_ch * hidden_size))
        self.seq = nn.Sequential(*seq_list)
        
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.0):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh, ch = [], [], []
        for i in range(nlayers):
            if i==0:
                ih.append(inner_cell(input_size, hidden_size, 2, 4))
                hh.append(inner_cell(hidden_size, hidden_size, 2, 4))
                ch.append(inner_cell(hidden_size, hidden_size, 2, 3))
            else:
                ih.append(inner_cell(hidden_size, hidden_size, 2, 4))
                hh.append(inner_cell(hidden_size, hidden_size, 2, 4))
                ch.append(inner_cell(hidden_size, hidden_size, 2, 3))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)
        self.w_ch = nn.ModuleList(ch)

    def forward(self, inputs, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](inputs) + self.w_hh[i](hx)
            gates1 = self.w_ch[i](cx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate1, f_gate1, o_gate1 = gates1.chunk(3, 1)
            i_gate = torch.sigmoid(i_gate + i_gate1)
            f_gate = torch.sigmoid(f_gate + f_gate1)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate + o_gate1)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * torch.tanh(ncx)
            cy.append(ncx)
            hy.append(nhx)
            inputs = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # [layer * batch * hidden]
        return hy, cy

class LSTMnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers=1):
        super(LSTMnet, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.nlayers = nlayers
        self.lstmcell = LSTMCell(input_size, hidden_size, nlayers, dropout = 0.0)
        self.linear = nn.Sequential(nn.Linear(hidden_size, 10*hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(10*hidden_size, output_size))
    def forward(self, inputs, hidden, npred = 0):
        output = []
        for i in range(inputs.size(0)):
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(self.nlayers, inputs.size(1), self.hidden_size, dtype=torch.double) #.cuda()
                    ct = torch.zeros(self.nlayers, inputs.size(1), self.hidden_size, dtype=torch.double) #.cuda()
                else:
                    ht, ct = hidden
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                output.append(self.linear(ht[-1]))
            else:
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                output.append(self.linear(ht[-1]))
                
            if i == npred:
                hidden = (ht, ct)
                
        return torch.stack(output, 0), hidden
    
class LSTMdense(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_size, nlayers=1):
        super(LSTMdense, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.lseq = seq_size
        self.lstmcell = LSTMCell(input_size, hidden_size, nlayers, dropout = 0.0)
        # add output linear combinations
        lin_list = []
        for i in range(seq_size):
            lin_list.append(nn.Sequential(nn.Linear(hidden_size, 5 * hidden_size),
                                    nn.ReLU(inplace = True),
                                    nn.Linear(5 * hidden_size, output_size)) )
        self.linear = nn.ModuleList(lin_list)
        
    def forward(self, inputs, hidden = (), npred = 0, device = "cpu"):
        output = []
        
        stage = []
        for i in range(inputs.size(0)):            
            if i == 0:
                if hidden == ():
                    ht = torch.zeros(self.nlayers, inputs.size(1), self.hidden_size, dtype=torch.double) #.to(device)
                    ct = torch.zeros(self.nlayers, inputs.size(1), self.hidden_size, dtype=torch.double) #.to(device)
                else:
                    ht, ct = hidden
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                # stage.append(ht[-1])
                
                # pred = f(ht[-1])
                # output.append(pred)
            else:
                ht, ct = self.lstmcell(inputs[i], (ht, ct))
                # stage.append(ht[-1])
                # pred += f(ht[-1])
                # output.append(pred)
                
            if i < self.lseq:
                stage.append(ht[-1])
            else:
                stage[:-1] = stage[1:]
                stage[-1] = ht[-1]
            if i >= self.lseq-1:
                pred = self.linear[-1](stage[-1]) #torch.zeros(inputs.size(1), self.output_size, dtype=torch.double).to(device)
                for l in range(self.lseq-1):
                    pred += self.linear[l](stage[l])
                output.append(pred)
                
            # if i == npred:
                # hidden = (ht, ct)
                
        return torch.stack(output, 0)  #, hidden

if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        seq = torch.randn(10, 100, 5, dtype=torch.double)
        lstm = LSTMdense(seq.size(2), 50, 4, 3).double()
        output_seq = lstm(seq)
        loss = output_seq.sum()
        loss.backward()
    print(output_seq.size())