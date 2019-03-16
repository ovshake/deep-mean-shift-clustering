import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self, input_channels, z_len):
		super(Encoder, self).__init__() 

		self.z_len = z_len
		self.input_channels = input_channels
		
		self.conv_1 = nn.Conv2d(self.input_channels, 20, kernel_size=5, stride=1)
		self.pool = nn.MaxPool2d(2, 2, return_indices=True)
		self.conv_2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
		self.fc_1 = nn.Linear(4*4*50, 500)
		self.fc_2 = nn.Linear(500, self.z_len)
		self.relu = nn.ReLU()
		
		self.ind1 = 0
		self.ind2 = 0

	def forward(self, x):
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

		return z , self.ind1 , self.ind2

if __name__ == '__main__':
	
	import torch

	z_len = 64

	batch_size = 216
	input_channels = 1
	inp_h = 28
	inp_w = 28

	x = torch.randn(batch_size, input_channels, inp_h, inp_w)

	encoder = Encoder(input_channels, z_len)

	z, ind1, ind2 = encoder(x)

	print('Z Vetcor', z.size())