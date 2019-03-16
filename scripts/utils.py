import mnist
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

class utils():

    def load_data(self, data_folder, train_batch_size, test_batch_size):
        
        # trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        transform = transforms.Compose([transforms.ToTensor()])

        train = torchvision.datasets.MNIST(data_folder, train=True, transform=transform, target_transform=None, download=True)
        test = torchvision.datasets.MNIST(data_folder, train=False, transform=transform, target_transform=None, download=True)

        train_loader = torch.utils.data.DataLoader(
                         dataset=train,
                         batch_size=train_batch_size,
                         shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
                        dataset=test,
                        batch_size=test_batch_size,
                        shuffle=True)

        return train_loader, test_loader

    def plot_regen(self, model, test_loader, num_im, save_folder, label, use_gpu):

        for batch_idx, (x, target) in enumerate(test_loader):

            image = x
            break

        # fig=plt.figure(figsize=(10,40))
        fig = plt.figure()
        plt.title(label)
        
        for i in range(num_im):
            data = np.array(image[i], dtype='float')
            data = data.reshape((28, 28))
            ax = fig.add_subplot(2, num_im, i+1)
            plt.imshow(data, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        data = image.float()
        if use_gpu:
            data = data.cuda() 
        
        out, embed = model(data)
        if use_gpu:
            out = out.cpu()

        out = out.detach().numpy()

        for i in range(num_im):
            out1 = out[i].reshape((28, 28))
            out1 = np.array(out1, dtype='float')
            
            ax = fig.add_subplot(2, num_im, num_im+i+1)
            plt.imshow(out1, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig(save_folder + 'test.png')

if __name__ == '__main__':
    
    train_batch_size = 216
    test_batch_size = 216
    data_folder = '~/btp_mean_shift/data/mnist/'

    util = utils()

    load_data = util.load_data
    plot_regen = util.plot_regen

    train_loader, test_loader = load_data(data_folder, train_batch_size, test_batch_size)
    print('Train Batches', len(train_loader))
    print('Test Batches', len(test_loader))

    from auto_encoder import AE as AE

    z_len = 64
    model = AE(z_len)

    num_im = 4
    use_gpu = False

    save_folder = '/home/ankitas/btp_mean_shift/save/test/'
    label = 'test'

    plot_regen(model, test_loader, num_im, save_folder, label, use_gpu)


