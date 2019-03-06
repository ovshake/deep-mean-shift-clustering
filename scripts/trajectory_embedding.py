import torch
import torch.nn as nn

class Path_LSTM(nn.Module):

    def __init__(self, z_len, l_len, use_gpu, num_layers=1, bias=True, bidirectional=False, dropout=0):
        super(Path_LSTM , self).__init__() 
        
        self.z_len = z_len
        self.l_len = l_len
        self.use_gpu = use_gpu
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.num_directions = int(self.bidirectional) + 1
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.z_len , hidden_size=self.l_len, batch_first=True, num_layers=self.num_layers, 
            bias=self.bias, bidirectional=self.bidirectional, dropout=self.dropout)
        if self.use_gpu:
            self.lstm = self.lstm.cuda()

    def forward(self, trajects):
        #trajects: batch_size X ms_iter X z_len
        batch_size = trajects.size()[0]
        
        h0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.l_len)
        c0 = torch.randn(self.num_layers*self.num_directions, batch_size, self.l_len)

        if self.use_gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()

        state = (h0, c0) 
        
        self.lstm.flatten_parameters()  #To avoid CUDA out of memory as LSTM on large data -> Too many params
        out, (h_n,c_n) = self.lstm(trajects, state) 
        # h_n: batch_size X num_layers*num_directions X l_len        

        l = torch.mean(h_n, dim=1, keepdim=False)        
        # l: batch_size X l_len

        return l

if __name__ == '__main__':
    
    batch_size = 216 
    ms_iter = 10
    z_len = 64
    l_len = 14
    use_gpu = torch.cuda.is_available()

    path_embed = Path_LSTM(z_len, l_len, use_gpu)

    trajects = torch.randn(batch_size, ms_iter, z_len)

    l = path_embed(trajects)

    print(l.size())
