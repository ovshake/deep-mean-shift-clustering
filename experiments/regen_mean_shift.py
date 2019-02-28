
# coding: utf-8

# # Deep Mean Shift Clustering

# ## Reqs

# ### Download

# In[1]:


# # http://pytorch.org/
# from os.path import exists
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
# accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

# !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
# !pip install pillow==4.1.1
# %reload_ext autoreload
# %autoreload
# !pip install python-mnist
# # !/usr/local/bin/python -m pip install visdom
# # !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip


# ### Import

# In[2]:


import mnist
import pickle
import torch
import visdom
vis = visdom.Visdom(port='8097')
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


# ## Dataset

# ### Download

# In[3]:


# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([transforms.ToTensor()])
root = '~/btp_mean_shift/content'
train = torchvision.datasets.MNIST(root, train=True, transform=trans, target_transform=None, download=True)
test = torchvision.datasets.MNIST(root, train=False, transform=trans, target_transform=None, download=True)


# ### Load

# In[4]:


train_batch_size = 512
test_batch_size = 100
train_loader = torch.utils.data.DataLoader(
                 dataset=train,
                 batch_size=train_batch_size,
                 shuffle=False)
test_loader = torch.utils.data.DataLoader(
                dataset=test,
                batch_size=test_batch_size,
                shuffle=False)

print ('total trainning batch number: '+ str(len(train_loader)))
print ('total testing batch number: '+ str(len(test_loader)))


# # AE Def

# ## Plotter

# In[5]:


def plot_regen(model):

    for batch_idx, (x, target) in enumerate(train_loader):

        train_image = x
        break

    fig=plt.figure(figsize=(10,40))

    num_im = 4
    for i in range(num_im):
        data = np.array(train_image[i], dtype='float')
        data = data.reshape((28, 28))
        ax = fig.add_subplot(1, num_im, i+1)
        plt.imshow(data, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

    fig=plt.figure(figsize=(10,40))

    data = train_image.float()
    out, embed = model(data.cuda())
    out = out.cpu().detach().numpy()

    num_im = 4
    for i in range(num_im):
        out1 = out[i].reshape((28, 28))
        out1 = np.array(out1, dtype='float')
        
        ax = fig.add_subplot(1, num_im, i+1)
        plt.imshow(out1, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

# plot_regen(model)


# ## Mean Shift Cluster

# In[6]:


class Mean_Shift_Cluster(torch.nn.Module):

    def __init__(self, delta, eta, ms_iter):
        super(Mean_Shift_Cluster, self).__init__()
        self.delta = delta
        self.eta = eta
        self.ms_iter = ms_iter

    def mean_shift_once(self, X):
        S = torch.mm(X.t(), X)
        K = torch.exp(self.delta * S)
        N = list(X.size())[1]
        d = torch.mm(K.t(), torch.ones(N, 1).cuda())
        q = 1 / d
        D_inv = torch.diagflat(q)
        eye = torch.eye(N).cuda()
        P = ((1-self.eta) * eye) + (self.eta * torch.mm(K, D_inv))
        return torch.mm(X, P)
  
    def forward(self, X):

        clust_embs = [0] * self.ms_iter

        clust_embs[0] = self.mean_shift_once(X)

        for it in range(1, self.ms_iter):
            clust_embs[it] = self.mean_shift_once(clust_embs[it-1])
    
        return clust_embs


# ## Loss

# In[7]:


class Loss(torch.nn.Module):

    def __init__(self, alpha):
        super(Loss, self).__init__()
        self.alpha = alpha

    def cluster_loss(self, embeddings, y):

        num_classes = len(np.unique(y)) 
        num_samples_for_each_class = [0 for i in range(num_classes)] 
        for i in y:
            num_samples_for_each_class[int(i)] += 1

        total = len(y)
        loss_list = torch.zeros(total, total).cuda()
        cos_sim = torch.nn.CosineSimilarity(dim=0)

        for i in range(total):
            for j in range(i+1, total):
                w_i = 1 / num_samples_for_each_class[y[i]] 
                w_j = 1 / num_samples_for_each_class[y[j]]
                scale = embeddings[i].norm(p=2) * embeddings[j].norm(p=2)
                if y[i] == y[j]:
                    loss_list[i,j] = (w_i * w_j) * (1 - cos_sim(embeddings[i] , embeddings[j])) * scale  
                else:
                    loss_list[i,j] = (w_i * w_j) * (torch.clamp(cos_sim(embeddings[i] , embeddings[j]) - alpha , 0)) * scale

        return loss_list.sum()/total
  
    def regen_loss(self, x, x_):
    
        ae_loss = torch.nn.MSELoss()
        regen_loss = torch.sum(torch.mm((x-x_).view(1, -1), (x-x_).view(-1, 1)))/x.view(-1).size()[0]
        return regen_loss
    
    def forward(self, X, X_, embeddings, y):
    
        ms_loss = self.cluster_loss(embeddings[0].t(), y)
        for i in range(1, len(embeddings)):
            ms_loss = ms_loss + self.cluster_loss(embeddings[i].t(), y)
            
        ae_loss = self.regen_loss(X, X_)

        return ms_loss + ae_loss, ms_loss, ae_loss


# ## Weigh init

# In[8]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)


# ## Model

# In[9]:


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

#     def mean_shift_cluster(self, X, delta, eta):
#         S = torch.mm(X.t(), X)
#         K = torch.exp(delta * S)
#         N = list(X.size())[1]
#         d = torch.mm(K.t(), torch.ones(N, 1).cuda())
#         q = 1 / d
#         D_inv = torch.diagflat(q)
#         eye = torch.eye(N).cuda()
#         P = ((1-eta) * eye) + (eta * torch.mm(K, D_inv))
#         return torch.mm(X, P)

    def decode(self, z):
        x = self.upfc_1(z)
        x = self.relu(x)
        x = self.upfc_2(x)
        x = x.view(x.size(0), 50, 4, 4)
        x = self.unpool(x, self.ind2)
        x = self.relu(x)
        x = self.upconv_1(x)
        x = self.unpool(x, self.ind1)
        x = self.relu(x)
        x = self.upconv_2(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)
        return x_, z


# # Model Init

# ## Params

# In[10]:


alpha = 0.5 
eta = 1 
ms_iter = 3
delta = 3 / (1-alpha)
total_epochs = 150
embedding_dim = 64


# ## Init

# In[11]:


model = AE(embedding_dim)
model.apply(weights_init)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.SGD(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

ms_clust = Mean_Shift_Cluster(delta, eta, ms_iter)

ms_ae_loss = Loss(alpha)
# new_loss = torch.nn.BCELoss(reduce=False)


# # Train

# In[ ]:


# model = AE(embedding_dim)
# model.load_state_dict(torch.load('/content/gdrive/My Drive/mean shift data/ms_ae_shift.pt'))
# model = model.cuda()

epoch_avg_total_loss = [0]*total_epochs

epoch_avg_cluster_loss = [0]*total_epochs

epoch_avg_regen_loss = [0]*total_epochs

total_loss_win = vis.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='epoch',title='Regen MS 3',ylabel='Total Loss',legend=['Loss']))

cluster_loss_win = vis.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='epoch',title='Regen MS 3',ylabel='Cluster Loss',legend=['Loss']))

regen_loss_win = vis.line(
    Y=np.zeros((1)),
    X=np.zeros((1)),
    opts=dict(xlabel='epoch',title='Regen MS 3',ylabel='Regen Loss',legend=['Loss']))

for epoch in range(total_epochs):
  
    print('epoch', epoch)
  
    avg_total_loss = 0
    avg_cluster_loss = 0
    avg_regen_loss = 0
    total_batches = 0
    
    for batch_idx, (x, target) in enumerate(train_loader):
        
        x, target = x.cuda(), target.numpy()

        x_, z = model(x)
        
        clust_embs = ms_clust(z.t()) 
        
        total_loss, cluster_loss, regen_loss = ms_ae_loss(x, x_, clust_embs, target)
    
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
        print('epoch', epoch, 'batch', batch_idx, 'batch total loss', total_loss.cpu().detach().numpy())
        print('epoch', epoch, 'batch', batch_idx, 'batch cluster loss', cluster_loss.cpu().detach().numpy())
        print('epoch', epoch, 'batch', batch_idx, 'batch regen loss', regen_loss.cpu().detach().numpy())
    
        avg_total_loss += total_loss.cpu().detach().numpy()
        avg_cluster_loss += cluster_loss.cpu().detach().numpy()
        avg_regen_loss += regen_loss.cpu().detach().numpy()
    
        if batch_idx % 5 == 0:
            print('saving...')
            torch.save(model.state_dict(), '/home/ankitas/btp_mean_shift/save/regen_shift_3.pt')
            print()
            plot_regen(model)
    
        total_batches += 1
   
    epoch_avg_total_loss[epoch] = avg_total_loss/total_batches
    epoch_avg_cluster_loss[epoch] = avg_cluster_loss/total_batches
    epoch_avg_regen_loss[epoch] = avg_regen_loss/total_batches
  
    loss_file = open('/home/ankitas/btp_mean_shift/save/loss_list_regen_shift_3.pkl', 'wb+')
    pickle.dump([epoch_avg_total_loss, epoch_avg_cluster_loss, epoch_avg_regen_loss], loss_file)
#     pickle.dump(epoch_avg_total_loss, loss_file)
    loss_file.close()
  
    print()
    print('epoch', epoch, 'epoch avg loss', epoch_avg_total_loss[epoch])
    print('epoch', epoch, 'epoch avg cluster loss', epoch_avg_cluster_loss[epoch])
    print('epoch', epoch, 'epoch avg regen loss', epoch_avg_regen_loss[epoch])
    print()

    vis.line(X=np.ones(1)*(epoch+1),Y=np.array([epoch_avg_total_loss[epoch]]),win=total_loss_win,update='append')
    vis.line(X=np.ones(1)*(epoch+1),Y=np.array([epoch_avg_cluster_loss[epoch]]),win=cluster_loss_win,update='append')
    vis.line(X=np.ones(1)*(epoch+1),Y=np.array([epoch_avg_regen_loss[epoch]]),win=regen_loss_win,update='append')
    plot_regen(model)
    print('saving...')
    torch.save(model.state_dict(), '/home/ankitas/btp_mean_shift/save/regen_shift_3.pt')
    print()
 
  

