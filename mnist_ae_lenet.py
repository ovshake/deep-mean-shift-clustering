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
test_batch_size = 100
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
		self.conv_1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
		self.pool = nn.MaxPool2d(2, 2, return_indices=True)
		self.conv_2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
		self.fc_1 = nn.Linear(4*4*50, 500)
		self.fc_2 = nn.Linear(500, 10)
		self.relu = nn.ReLU()
		self.ind1 = 0
		self.ind2 = 0

		self.upfc_1 = nn.Linear(10, 500)
		self.upfc_2 = nn.Linear(500, 4*4*50)
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.upconv_1 = nn.ConvTranspose2d(50, 20, kernel_size=5, stride=1)
		self.upconv_2 = nn.ConvTranspose2d(20, 1, kernel_size=5, stride=1) 

	def encode(self,x):
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
		# z_norm = z.norm(p=2, dim=1, keepdim=True).detach()
		# z = z.div(z_norm.expand_as(z))
		return z 

	def decode(self , x):
		x = self.upfc_1(x) 
		x = self.relu(x)
		x = self.upfc_2(x)
		x = x.view(x.size(0), 50, 4, 4)
		x = self.unpool(x, self.ind2)
		x = self.relu(x)
		x = self.upconv_1(x)
		x = self.unpool(x, self.ind1)
		x = self.relu(x)
		x = self.upconv_2(x)
		return x

	def forward(self,z):
		z = self.encode(z) 
		z = self.decode(z) 
		return z 



def batch_plot(batch, reconstructed , n_plots = 4):
	fig,ax = plt.subplots(nrows = 2, ncols = n_plots)
	for i in range(n_plots):
		x = batch[i].view(28,28) 
		x_hat = reconstructed[i].view(28,28) 
		ax[0 , i ].imshow(x.cpu().detach()  ,cmap = 'gray')
		ax[1 , i].imshow(x_hat.cpu().detach()  ,cmap = 'gray')



model = AE() 
model = model.cuda() 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
msae_loss = nn.MSELoss() 

save_dir = 'models/'

num_epochs = 50
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

	if n_epoch % 10 == 0:
		for batch_idx, (x,target) in enumerate(test_loader):
			x , target = x.cuda() , target.cuda() 
			x_hat = model(x) 
			batch_plot(x , x_hat) 
			print('ploting....')
			plt.savefig('plots/'+ str(n_epoch) + '_testimages.jpeg') 
			plt.close()  
			break 


