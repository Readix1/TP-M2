import torch.nn as nn
import torch
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, dimin, dimout, layers, l1=0,l2=0, d=0, norm=False):
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
        
        for i,la in enumerate(layers):
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
