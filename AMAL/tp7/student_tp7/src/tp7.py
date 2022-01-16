import logging

#from tp8.student_tp7.src.lightningexemple import BATCH_SIZE
#logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, dataloader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from collections import OrderedDict
import datetime
from torch.distributions import Categorical
from math import log

from datamaestro import prepare_dataset


class CustomImageDataset(Dataset):
    def __init__(self, labels, x):
        self.y = torch.tensor(labels).long()
        self.x = torch.tensor(x).flatten(1).float()/255

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        label = self.y[idx]
        return x, label

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
BATCHSIZE = 300
EPOCH = 1000

ds = prepare_dataset ( "com.lecun.mnist" )
# Ne pas  o u b l i e r  se  sous−é chantillonner  !
trainV_img, trainV_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

g = torch.Generator()
g.manual_seed(12345)

ind = torch.randperm(trainV_img.shape[0], generator=g)
indTrain = ind[:int(trainV_img.shape[0]*TRAIN_RATIO)]
indVal = ind[int(trainV_img.shape[0]*TRAIN_RATIO):int(trainV_img.shape[0]*TRAIN_RATIO)+1000]

train_img, train_labels = trainV_img[indTrain], trainV_labels[indTrain]
val_img, val_labels = trainV_img[indVal], trainV_labels[indVal]

training_data = CustomImageDataset(train_labels, train_img)
val_data = CustomImageDataset(val_labels, val_img)
test_data = CustomImageDataset(test_labels, test_img)

train_dataloader = DataLoader(training_data, batch_size=BATCHSIZE, shuffle=True,drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCHSIZE, shuffle=True)

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


class SeqPerso(nn.Module):
    def __init__(self, dimin, dimout, layer, l1=0,l2=0, d=0, norm=False):
        """
        """
        super().__init__()
        self.l1=l1
        self.l2=l2
        self.dimin = dimin 
        self.dimout = dimout
        self.layer = []
        self.track = []
        self.linears = []
        currentIn = dimin
        
        for i,la in enumerate(layer):
            l = nn.Linear(currentIn, la)
            self.layer.append(("Linear"+str(i),l))
            self.linears.append(l)
            if d >0:
                self.layer.append(("Drop"+str(i),nn.Dropout(d)))
            if norm:
                self.layer.append(("BatchNorm"+str(i),nn.BatchNorm1d(la)))
            self.layer.append(("ReLU"+str(i),nn.ReLU()))
            
            currentIn=la
        l = nn.Linear(currentIn, dimout)
        self.layer.append(("Linear out",l))
        self.linears.append(l)
        
        self.seq  = torch.nn.Sequential(OrderedDict(self.layer))
        
    def forward(self, x):
        return self.seq(x)
    
    def computeL1(self):
        sum=0
        for l in self.linears:
            sum += torch.norm(l.weight, 1)
        return self.l1*sum
    
    def computeL2(self):
        sum=0
        for l in self.linears:
            sum += torch.norm(l.weight, 2)**2
        return self.l2*sum

def grid_search(params,param=dict()):
    if(not params):
        train(param)
        return 
    
    k,list_v = next(iter(params.items()))
    params2 = params.copy()
    params2.pop(k)
    for v in list_v:
        param2 = param.copy()
        param2[k] = v
        grid_search(params2,param2)
    

def train(params,overwrite=False):
    nom = "_".join(sorted([str(k)+"="+str(v) for k,v in params.items()]))
    p = 'runs/tp7/'+nom
    if os.path.isdir(p):
        if(overwrite):
            err=os.system('rmdir /S /Q "{}"'.format(p))
        else : 
            return 
    writer = SummaryWriter(p)
    
    modele = SeqPerso(28*28, 10, [100,100,100], **params)   
    loss = nn.CrossEntropyLoss() 
    optim = torch.optim.SGD(modele.parameters(), lr=0.1)

    for epoch in range(EPOCH):
        l=0
        n=0
        entropy = torch.Tensor()
        modele.train()
        for x,y in train_dataloader:
            optim.zero_grad()
            pred = modele(x)
            lo = loss(pred,y)
            l+=lo
            lo += modele.computeL1()+modele.computeL2()
            if epoch%50==0:
                entropy = torch.cat((entropy,Categorical(logits= pred).entropy()))
                for m in range(len(modele.linears)):
                    store_grad(modele.linears[m].weight)
            lo.backward()
            optim.step()
            n+=1
        writer.add_scalar('Loss/train', l/n, epoch)
        #writer.add_scalar('Accuracy/train', np.random.random(), i)*
        poids = None
        grad=None
        if epoch%50==0:
            poids = torch.Tensor()
            grad = torch.Tensor()
            for m in range(len(modele.linears)):
                poids = torch.cat((poids, modele.linears[m].weight.flatten()))
                grad = torch.cat((grad, modele.linears[m].weight.grad.flatten()))
            writer.add_histogram("Linear weight", poids, epoch)
            writer.add_histogram("Linear grad", grad, epoch)
            writer.add_histogram("Entropy (log10)", entropy/log(10), epoch)
        
        l=0
        n=0
        nbex=0
        modele.eval()
        acc=0
        with torch.no_grad():
            for x,y in val_dataloader:
                pred = modele(x)
                l += loss(pred,y)
                n+=1
                nbex+=len(y)
                acc+= torch.where(torch.argmax(pred,dim=1)==y,1,0).sum()
        writer.add_scalar('Loss/validation', l/n, epoch)
        writer.add_scalar('Accuracy/validation',acc/nbex, epoch)

grid_search({'l1':[0.],'l2':[0.01],'d':[0],'norm':[False,True]})