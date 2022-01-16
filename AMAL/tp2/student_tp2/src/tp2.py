import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm




data=datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datax = datax/torch.max(datax)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)


xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.2)

def linearBatch():
    writer = SummaryWriter()
    w = torch.rand((13,1),requires_grad=True)
    b = torch.rand((1,1),requires_grad=True)
    alpha = 1e-2

    with torch.no_grad():
        z=xtrain.mm(w) + b
        loss = ((ytrain-z)**2).sum()/ytrain.size(0)
        print(loss)

    for i in range(1000):
        z = xtrain.mm(w) + b
        loss = ((ytrain-z)**2).sum()/ytrain.size(0)
        loss.backward()
        with torch.no_grad():
            w = w - alpha*w.grad #ou w.data
            b = b - alpha*b.grad
        w.requires_grad=True
        b.requires_grad=True
        writer.add_scalar('Loss/train', loss, i)#tensorboard --logdir runs
        with torch.no_grad():
            z=xtrain.mm(w) + b
            loss = ((ytrain-z)**2).sum()/ytrain.size(0)
            writer.add_scalar('Loss/test', loss, i)

    with torch.no_grad():
        z=xtrain.mm(w) + b
        loss = ((ytrain-z)**2).sum()/ytrain.size(0)
        print(loss)


def linearTorch():
    writer = SummaryWriter()
    alpha = 1e-2
    NB_EPOCH = 1000

    seq = torch.nn.Sequential(torch.nn.Linear(13,20), torch.nn.Tanh(), torch.nn.Linear(20,1))
    loss = torch.nn.MSELoss(reduction="mean")

    optim = torch.optim.SGD(params=seq.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()
    # Reinitialisation du gradient
    for i in range(NB_EPOCH):
        l = loss.forward(ytrain, seq.forward(xtrain)) #Calcul du cout
        l.backward() # Retropropagation
        optim.step() # Mise-à-jour des paramètres w et b
        optim.zero_grad() # Reinitialisation du gradient
        writer.add_scalar('Loss/train', l, i)#tensorboard --logdir runs
        if i%100==0:
            print((i/NB_EPOCH)*100,'%')
        with torch.no_grad():
            l = loss.forward(ytest, seq.forward(xtest))
            writer.add_scalar('Loss/test', l, i)



def anciencode():
    """ 
    a = torch.rand((1,10),requires_grad=True)
    b = torch.rand((1,10),requires_grad=True)
    c = a.mm(b.t())
    d = 2 * c
    c.retain_grad() # on veut conserver le gradient par rapport à c
    d.backward() ## calcul du gradient et retropropagation
    ##jusqu’aux feuilles du graphe de calcul
    print(d.grad) #Rien : le gradient par rapport à d n’est pas conservé
    print(c.grad) # Celui-ci est conservé
    print(a.grad) ## gradient de d par rapport à a qui est une feuille
    print(b.grad) ## gradient de d par rapport à b qui est une feuille
    d = 2 * a.mm(b.t())
    d.backward()
    print(a.grad) ## 2 fois celui d’avant, le gradient est additioné
    a.grad.data.zero_() ## reinitialisation du gradient pour a
    d = 2 * a.mm(b.t())
    d.backward()
    print(a.grad) ## Cette fois, c’est ok
    with torch.no_grad():
        c = a.mm(b.t()) ## Le calcul est effectué sans garder le graphe de calcul
    c.backward() ## Erreur"""


