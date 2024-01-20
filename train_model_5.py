import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
from utils import *
from GS_functions import GF
import sys

from modeling.model_5 import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#########################################################################################

start_neuron, end_neuron = int(sys.argv[1]),int(sys.argv[2]) #non inclusive
num_neurons = end_neuron-start_neuron
site = sys.argv[3]
print("Model 5\n")
print(f'Start, End Neuron: {start_neuron}, {end_neuron}\n')
print(f'Site: {site}')

train_img = np.load(f'../all_sites_data_prepared/pics_data/train_img_{site}.npy')
train_resp = np.load(f'../all_sites_data_prepared/New_response_data/trainRsp_{site}.npy')
train_img=np.reshape(train_img,(-1,1,50,50))

train_resp = train_resp[:,start_neuron:end_neuron]

train_dataset = ImageDataset(train_img,train_resp,num_neurons)
train_loader = DataLoader(train_dataset,50,shuffle=True)

val_img = np.load(f'../all_sites_data_prepared/pics_data/val_img_{site}.npy')
val_resp = np.load(f'../all_sites_data_prepared/New_response_data/valRsp_{site}.npy')
val_img = np.reshape(val_img,(-1,1,50,50))

val_resp = val_resp[:,start_neuron:end_neuron]
val_dataset = ImageDataset(val_img,val_resp,num_neurons)
val_loader = DataLoader(val_dataset,50,shuffle=True)

#########################################################################################

MAXCORRSOFAR = [0]*num_neurons
MAXAVGCORRSOFAR = 0

savepathname=f'{site}/model_5'
GF.mkdir('./results/',savepathname)
for i in range(num_neurons):
    nfold = "neuron_" + str(i+start_neuron)
    GF.mkdir("./results/" + savepathname,nfold)
GF.mkdir("./results/" + savepathname,"avg")
log = open(f'./results/{savepathname}/log.txt', "w")

net = model_5(num_neurons = num_neurons)
opt = torch.optim.Adam(net.parameters(),lr=0.001)
lfunc = torch.nn.MSELoss()

net.to(device)

#########################################################################################

losses = []
accs = []
for epoch in range(50): 
    net.train()
    train_losses=[]
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)
        l = lfunc(net(x),y)
        opt.zero_grad()
        l.backward()
        opt.step()
        train_losses.append(l.item())
    losses.append(np.mean(train_losses))
    
    val_losses=[]
    with torch.no_grad():
        net.eval()
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            l = lfunc(net(x),y)
            val_losses.append(l.item())
    accs.append(np.mean(val_losses))

    log.write("Epoch " + str(epoch) + " train loss is:" + str(losses[-1])+"\n")
    log.write("Epoch " + str(epoch) + " val loss is:" + str(accs[-1])+"\n")
    print("Epoch " + str(epoch) + " train loss is:" + str(losses[-1])+"\n")
    print("Epoch " + str(epoch) + " val loss is:" + str(accs[-1])+"\n")

    mean_corrs,corrs =  pc_firstx_neuron(net,num_neurons,val_img,val_resp,device)

    log.write("Epoch " + str(epoch) + " mean correlation is:" + str(mean_corrs)+"\n")
    log.write("Epoch " + str(epoch) + " correlations:" + str(corrs)+"\n")
    print("Epoch " + str(epoch) + " mean correlation is:" + str(mean_corrs)+"\n")
    print("Epoch " + str(epoch) + " correlations:" + str(corrs)+"\n")
    
    for i in range(num_neurons):
        if corrs[i] > MAXCORRSOFAR[i]:
            MAXCORRSOFAR[i] = corrs[i]
            log.write("Epoch " + str(epoch) + ": model neuron " + str(start_neuron+i) + " saved sucessfully.\n")
            print("Epoch " + str(epoch) + ": model neuron " + str(start_neuron+i) + " saved sucessfully.\n")
            torch.save(net.models[i].state_dict(),  f'./results/{savepathname}/neuron_{start_neuron+i}/model_epoch_{epoch}.pth')

    if mean_corrs > MAXAVGCORRSOFAR:
        MAXAVGCORRSOFAR = mean_corrs
        log.write("Epoch " + str(epoch) + ": saved sucessfully.\n")
        print("Epoch " + str(epoch) + ": saved sucessfully.\n")
        torch.save(net.state_dict(),  f'./results/{savepathname}/avg/model_epoch_{epoch}.pth')

    log.write("\n")
    print("\n")

log.close()
