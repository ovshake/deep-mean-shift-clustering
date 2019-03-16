import torch.nn as nn

class Decoder(nn.Module):
	def __init__(self, input_channels, z_len):
		super(Decoder, self).__init__()
		
		self.z_len = z_len
		self.input_channels = input_channels

		self.upfc_1 = nn.Linear(self.z_len, 500)
		self.upfc_2 = nn.Linear(500, 4*4*50)
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.upconv_1 = nn.ConvTranspose2d(50, 20, kernel_size=5, stride=1)
		self.upconv_2 = nn.ConvTranspose2d(20, self.input_channels, kernel_size=5, stride=1) 

		self.relu = nn.ReLU()
	
	def forward(self, z, ind1, ind2):
		x = self.upfc_1(z) 
		x = self.relu(x)
		x = self.upfc_2(x)
		x = x.view(x.size(0), 50, 4, 4)
		x = self.unpool(x, ind2)
		x = self.relu(x)
		x = self.upconv_1(x)
		x = self.unpool(x, ind1)
		x = self.relu(x)
		x = self.upconv_2(x)
		return x

if __name__ == '__main__':
	
	from encoder import Encoder as Encoder

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

	decoder = Decoder(input_channels, z_len)

	x = decoder(z, ind1, ind2)

	print('Regen X vector', x.size())
