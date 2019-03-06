import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, z_len):
        super(AE, self).__init__()

        self.conv_1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.ind1 = 0
        self.ind2 = 0
        self.conv_2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(4*4*50, 500)
        self.fc_2 = nn.Linear(500,z_len)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.upfc_1 = nn.Linear(z_len, 500)
        self.upfc_2 = nn.Linear(500, 4*4*50)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.upconv_1 = nn.ConvTranspose2d(50, 20, kernel_size=5, stride=1)
        self.upconv_2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1)  

    def encode(self, x):
        z = self.conv_1(x)
        z = self.relu(z)
        z, self.ind1 = self.pool(z)
        z = self.conv_2(z)
        z = self.relu(z)
        z, self.ind2 = self.pool(z)
        z = z.view(z.size(0), -1)
        z = self.fc_1(z)
        z = self.relu(z)
        z = self.fc_2(z)
        z_norm = z.norm(p=2, dim=1, keepdim=True).detach()
        z = z.div(z_norm.expand_as(z))
        return z

    def decode(self, z):
        x_ = self.upfc_1(z)
        x_ = self.relu(x_)
        x_ = self.upfc_2(x_)
        x_ = x_.view(x_.size(0), 50, 4, 4)
        x_ = self.unpool(x_, self.ind2)
        x_ = self.relu(x_)
        x_ = self.upconv_1(x_)
        x_ = self.unpool(x_, self.ind1)
        x_ = self.relu(x_)
        x_ = self.upconv_2(x_)
        x_ = self.sigmoid(x_)
        return x_

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)
        return x_, z

if __name__ == '__main__':   
    
    batch_size = 216
    inp_channels = 1
    inp_h = 28
    inp_w = 28
    
    z_len = 64
    
    model = AE(z_len)

    x = torch.randn(batch_size, inp_channels, inp_h, inp_w)
    
    print(x.size())
    
    x_, z = model(x)
    
    print(x_.size(), z.size())
    
    