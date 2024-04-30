import torch
import torch.nn as nn

__all__ = ['LSTM']

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, inputs):
        hidden_output, (h_n, c_n) = self.lstm(inputs)
        final_out = []
        for i in range(hidden_output.size(0)):
            final_out.append(self.linear(hidden_output[i]))
        return torch.stack(final_out, 0)

if __name__ == '__main__':
    lstm = LSTM(5, 10, 4, 2)
    seq = torch.randn(2000, 100, 5)
    output_seq = lstm(seq)
    print(output_seq.size())