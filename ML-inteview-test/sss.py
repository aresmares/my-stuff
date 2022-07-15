from cProfile import run
from inspect import trace
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

## Design your network
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

# import torchvision
# import tochvision.transforms as transforms

class CNN(nn.Module):
  def __init__(self, inchannels = 3, num_classes = 1, batch_size=4):
    super(CNN, self).__init__()
    # SAME CONVOLUTION - output after conv2d will remain the same; dimensionalty is not reduced
    self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3, stride=1, padding=1)

    self.fc1 = nn.Linear(in_features=16*16*16, out_features=64*64)
    self.fc2 = nn.Linear(in_features=64*64, out_features=64*64)

    self.conv_red = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)     # reduce dimensionality           
    self.sigmoid = nn.Sigmoid()                                                                                                                                                          


  def forward(self, x):
    # first conv layer
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)
    
    # second conv layer
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)

    # reshape tensor
    x = x.reshape(x.shape[0], -1)

    # fully connected layer
    x = self.fc1(x)
    # x = self.fc2(x)

    x = self.sigmoid(x)
    x = x.reshape((4,1,64,64))

    # x = torch.clamp(x,min=0, max=1)
    # x = torch.round(x)
    # # x = x.reshape((4,1,64,64))

    # plt.imshow( x[0].permute(1,2,0).detach().numpy() )

    return x

# model = CNN()
# inputs = torch.rand(4, 3, 64, 64)
# print(model(inputs).shape)

## Training parameters
CHAN_IN = 3
CHAN_OUT = 1
EPOCHS = 50
DATASET_LEN = 50
BATCH_SIZE = 4

PRINT_LOSS_FREQUENCY = 100

## We give a mock train_dataset which is a list of random input and 
## groundtruth data pairs: [[input1, mask1], [input2, mask2], etc.].
import numpy as np
inputs = [torch.rand(BATCH_SIZE, CHAN_IN, 64, 64) for x in range(DATASET_LEN)]
masks = [torch.round(torch.rand(BATCH_SIZE, CHAN_OUT, 64, 64)) for x in range(DATASET_LEN)]
train_dataset = [[inputs[idx], masks[idx]] for idx in range(DATASET_LEN)]

## Write a training loop that runs for EPOCHS and print the loss for every PRINT_LOSS_FREQUENCY steps
# For example, it could start like this:

model = CNN()
optimiser = optim.Adam(model.parameters()) # default := lr=0.001
loss = nn.MSELoss()

steps = 0
running_loss=0
training_loss = []

fmask = None
fscore = None

print("[*] Start Training")
for epoch in range(EPOCHS):
    for input, mask in train_dataset:
        
        # plt.figure();  plt.title("Pre- mask"); plt.imshow( mask[0].permute(1,2,0).detach().numpy() )
        # plt.figure();  plt.title("Pre- Inp"); plt.imshow( input[0].permute(1,2,0).detach().numpy() )

        # forward pass
        scores = model(input)

        l_value = loss(scores, mask)
        training_loss.append(l_value.item())

        running_loss += l_value.item()

        # back pass
        optimiser.zero_grad()
        l_value.backward()

        # use adam to update parameters
        optimiser.step()
        
        fmask = mask
        fscore = scores
        
        if steps%PRINT_LOSS_FREQUENCY == 0:
            train_loss = running_loss/len(train_dataset)
            print("Step: %d: Epoch %d, Training loss %.10f" % (steps, epoch, running_loss))   

        steps += 1


plt.plot(training_loss)
plt.figure(); plt.title("Post- Mask"); plt.imshow( fmask[0].permute(1,2,0).detach().numpy() )
plt.figure(); plt.title("Post- Score"); plt.imshow( fscore[0].permute(1,2,0).detach().numpy() )
plt.show()
