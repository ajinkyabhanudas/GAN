# discrimeep Convolutional genANs

#library imports########
from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.optim as optim
import torch.utils.data
#hyperparameters###########
batch_size = 128 # We set the size of the batch.
image_size = 128 # We set the size of the generated images (64x64).
transform = transforms.Compose([transforms.Scale(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.discrimataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# discrimefining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.03)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.03)
        m.bias.data.fill_(0)

# discrimefining the generator

class gen(nn.Module):

    def __init__(self):
        super(gen, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


net_grad = gen()
net_grad.apply(weights_init)


class discrim(nn.Module):

    def __init__(self):
        super(discrim, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Creating the discriminator
net_discrim = discrim()
net_discrim.apply(weights_init)

# Training genAN ###########

criterion = nn.BCELoss()
optimizerdiscrim = optim.Adam(net_discrim.parameters(), lr = 0.0002, betas = (0.5, 0.998))
optimizergen = optim.Adam(net_grad.parameters(), lr = 0.0002, betas = (0.5, 0.997))

for epoch in range(25):

    for i, data in enumerate(dataloader, 0):
       
        net_discrim.zero_grad()
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = net_discrim(input)
        error_real = criterion(output, target)
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = net_grad(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = net_discrim(fake.detach())
        error_fake = criterion(output, target)
        
        
        error = error_real + error_fake
        error.backward()
        optimizerdiscrim.step()
        net_grad.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = net_discrim(fake)
        error_grad = criterion(output, target)
        error_grad.backward()
        optimizergen.step()

        print('[%d/%d][%d/%d] Loss_discrim: %.4f Loss_gen: %.4f' % (epoch, 25, i, len(dataloader), error.data[0], error_grad.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = net_grad(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)