import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable, Function, gradcheck
import matplotlib.pyplot as plt 

trans = transforms.Compose([transforms.ToTensor()]) 
# os.mkdir('data/') 
root = 'data/'
train = torchvision.datasets.MNIST(root, train=True, transform=trans, target_transform=None, download=True) 
test =  torchvision.datasets.MNIST(root, train=False, transform=trans, target_transform=None, download=True) 

train_batch_size = 100
test_batch_size = 1 
train_loader = torch.utils.data.DataLoader(
                 dataset=train,
                 batch_size=train_batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test,
                batch_size=test_batch_size,
                shuffle=True)



class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__() 
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, stride = 1) #resulting in 24x24x20
        self.conv_2 = nn.Conv2d(in_channels = 20, out_channels = 5, kernel_size = 3, stride = 1) # 22x22x5
        self.max_pool_1 = nn.MaxPool2d(kernel_size = 4, stride = 1 , return_indices = True) # 19x19x5 
        self.fc_1 = nn.Linear(19*19*5, 64 * 2) 
        self.relu_1 = nn.ReLU() 
        self.fc_2 = nn.Linear(64 * 2, 64) 
        self.up_fc1 = nn.Linear(64 , 64 * 2) 
        self.relu_2 = nn.ReLU() 
        self.up_fc2 = nn.Linear(64 * 2, 19 * 19 * 5)
        self.ind1 = 0
        self.ind2 = 0
        self.max_unpool_1 = nn.MaxUnpool2d(kernel_size = 4, stride = 1)  
        self.up_conv_1 = nn.ConvTranspose2d(in_channels = 5, out_channels = 20, kernel_size = 3, stride = 1) 
        self.up_conv_2 = nn.ConvTranspose2d(in_channels = 20, out_channels = 1 , kernel_size = 5, stride = 1) 

    def encode(self,x):
        z = self.conv_1(x) 
        z = self.conv_2(z) 
        z , self.ind1 = self.max_pool_1(z)
        z = z.view(z.size(0), -1)  #train_batch_size
        # print(z) 
        z = self.fc_1(z) 
        z = self.relu_1(z)
        z = self.fc_2(z) 
        return z 

    def decode(self , x):
        z = self.up_fc1(x)
        z = self.relu_2(z)  
        z = self.up_fc2(z) 
        z = z.view(z.size(0) , 5, 19, 19)
        z = self.max_unpool_1(z , self.ind1)  
        z = self.up_conv_1(z) 
        z = self.up_conv_2(z)
        return z 

    def forward(self,z):
        z = self.encode(z) 
        z = self.decode(z) 
        return z 






model = AE() 
model = model.cuda() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
msae_loss = nn.MSELoss() 

save_dir = 'models/'

num_epochs = 10
optimizer.zero_grad() 
for n_epoch in range(num_epochs):
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad() 
        x , target = x.cuda(), target.cuda() 
        x_hat = model(x) 
        loss = msae_loss(x, x_hat) 
        loss.backward() 
        optimizer.step()
        print('LOSS =====> {} , EPOCH ======> {} '.format(loss, n_epoch))  



for batch_idx, (x,target) in enumerate(test_loader):
    x , target = x.cuda() , target.cuda() 
    x_hat = model(x) 
    x = x.view(28,28) 
    x_hat = x_hat.view(28,28) 
    print(x.size() , x_hat.size()) 
    fig , ax = plt.subplots(nrows = 1, ncols = 2) 
    ax[0].imshow(x_hat.cpu().detach(),cmap = 'gray') 
    ax[1].imshow(x.cpu().detach() ,cmap = 'gray') 
    plt.show() 












