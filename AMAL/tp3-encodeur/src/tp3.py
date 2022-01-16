from pathlib import Path
import os
import torch
from torch._C import _disable_minidumps
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import datetime
import utils

# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = torch.tensor(ds.train.images.data()), torch.tensor(ds.train.labels.data())
test_images, test_labels =  torch.tensor(ds.test.images.data()), torch.tensor(ds.test.labels.data())

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
#writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
##images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
##images = make_grid(images)
# Affichage avec tensorboard
##writer.add_image(f'samples', images, 0)

################################### DATASET ###################################

class MonDataSet(Dataset):
    def __init__(self, x, y):
        self.y =y
        self.x =x.view(x.size(0),-1).float()/x.max()
    def __getitem__(self, index):
        return self.x[index], self.x[index]
    def __len__(self):
        return self.y.size(0)


def _init_fn(worker_id):
    np.random.seed(12 + worker_id)

seed = torch.Generator()
seed.manual_seed(0)

monDs = MonDataSet(train_images, train_labels)
dsTest = MonDataSet(test_images, test_labels)

data_train = DataLoader(monDs, batch_size=100, worker_init_fn = _init_fn, generator = seed)
data_test = DataLoader( dsTest,  batch_size=len(dsTest), worker_init_fn = _init_fn, generator = seed)


########################################################################################

###################################### AUTOENCODER ##################################################

class AE(nn.Module):
    def __init__(self, dim_im, dim_l):
        super().__init__()
        self.dim_im=dim_im
        self.dim_l=dim_l
        self.linear1 = nn.Linear(dim_im, dim_l, bias=True)
        self.biais = nn.Parameter(torch.rand(dim_im, requires_grad=True))
        

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        return torch.sigmoid(self.linear1(data))

    def decode(self,data):
        res = F.linear(data,self.linear1.weight.t(),self.biais)
        return torch.sigmoid(res)

############################ test ##################

ae = AE(len(monDs[0][0]),30)
ae = utils.optim(ae,data_train,torch.nn.MSELoss(reduction="mean"),epoch=100,alpha=1000,nameState='ae_30_lr_1000',affFunc = lambda s,lf : utils.aff_complet(s,lf,monDs,dsTest,name='ae_1couche_dim30_lr1000'))

x,y = monDs[:]
train_loss = utils.compute_loss(ae,x,y,torch.nn.MSELoss(reduction="mean"))
x,y = dsTest[:]
test_loss = utils.compute_loss(ae,x,y,torch.nn.MSELoss(reduction="mean"))
print(train_loss.item(),test_loss.item())

############################################################################################
############################## affichage de quelques exemples ##############################
############################################################################################
def aff_exemples(model,name):
    images_predict = []

    with torch.no_grad():
        for i in range(10):
            x,y=dsTest[i]
            #plt.imshow(x.view(28,28))
            #plt.show()
            x=x.to(device='cuda')
            im_p = model(x).view(28,28).to(device='cpu')
            images_predict.append(im_p)
            #plt.imshow(im_p)
            #plt.show()

        images_predict = torch.stack(images_predict)

    # Pour visualiser
    writer = SummaryWriter("runs/"+name)
    # Les images doivent etre en format Channel (3) x Hauteur x Largeur
    images_test = torch.tensor(test_images[0:10]).unsqueeze(1).repeat(1,3,1,1).double()/255.
    images_predict = images_predict.unsqueeze(1).repeat(1,3,1,1).double()
    # Permet de fabriquer une grille d'images
    images_test = make_grid(images_test)
    images_predict = make_grid(images_predict)
    print(images_test.size(),images_predict.size())
    # Affichage avec tensorboard
    writer.add_image(f'samples', images_test, 0)
    writer.add_image(f'predict', images_predict, 1)

aff_exemples(ae,'exemples_1couche_30')

############################################################################################
############################################################################################
############################################################################################

############################################################################################
########################### Mélange d'embeddings ###########################################
############################################################################################
writer = SummaryWriter("runs/melange_emb_27")

with torch.no_grad():
    i0 = torch.argmax(torch.where(test_labels==7,1,0))
    i2 = torch.argmax(torch.where(test_labels==2,1,0))
    img,_ = dsTest[i0]
    img2,_ = dsTest[i2]
    # plt.imshow(img.view(28,28).to(device='cpu'))
    # plt.show()
    # plt.imshow(img2.view(28,28).to(device='cpu'))
    # plt.show()
    img=img.to(device='cuda')
    img2=img2.to(device='cuda')
    emb = ae.encode(img)
    emb2 = ae.encode(img2)
    
    decodes = []
    for lamb in range(1,10):
        lamb/=10
        embres = lamb*emb+(1-lamb)*emb2
        decode = ae.decode(embres).view(28,28).to(device='cpu')
        decodes.append(decode)
    
    decodes = torch.stack(decodes)
    decodes = decodes.unsqueeze(1).repeat(1,3,1,1).double()
    decodes = make_grid(decodes)
    writer.add_image(f'samples', decodes, 0)

############################################################################################
############################################################################################
############################################################################################


############################################################################################
############################## multicouche #################################################
############################################################################################
class AEmulticouche(nn.Module):
    def __init__(self, listdim):
        super().__init__()
        self.dims=listdim
        self.n = len(self.dims)-1
        self.linears = nn.ModuleList([nn.Linear(self.dims[i],self.dims[i+1] , bias=True) for i in range(self.n)])
        self.biais = nn.ParameterList([nn.Parameter(torch.rand(self.dims[i], requires_grad=True)) for i in range(0,self.n)])
        
    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        out = data
        for i in range(self.n):
            out = torch.sigmoid(self.linears[i](out))
        return out

    def decode(self,data):
        out = data
        for i in range(self.n-1,-1,-1):
            out = torch.sigmoid(F.linear(out,self.linears[i].weight.t(),self.biais[i]))
        return out


###test ae multicouche
model = AEmulticouche([len(monDs[0][0]),500,100,30])
aemulti = utils.optim(model,data_train,torch.nn.MSELoss(reduction="mean"),epoch=100,alpha=100,nameState='modelmulti50010030_a_100',affFunc = lambda s,lf : utils.print_loss(s,lf,monDs,dsTest))

x,y = monDs[:]
train_loss = utils.compute_loss(aemulti,x,x,torch.nn.MSELoss(reduction="mean"))
x,y = dsTest[:]
test_loss = utils.compute_loss(aemulti,x,x,torch.nn.MSELoss(reduction="mean"))
print(train_loss.item(),test_loss.item())

############################################################################################
############################## affichage de quelques exemples ##############################
############################################################################################
aff_exemples(aemulti,'exemples_multicouche_dim30')

############################################################################################
########################### Mélange d'embeddings ###########################################
############################################################################################
writer = SummaryWriter("runs/melange_multicouche_emb_27")

with torch.no_grad():
    i0 = torch.argmax(torch.where(test_labels==7,1,0))
    i2 = torch.argmax(torch.where(test_labels==2,1,0))
    img,_ = dsTest[i0]
    img2,_ = dsTest[i2]
    # plt.imshow(img.view(28,28).to(device='cpu'))
    # plt.show()
    # plt.imshow(img2.view(28,28).to(device='cpu'))
    # plt.show()
    img=img.to(device='cuda')
    img2=img2.to(device='cuda')
    emb = aemulti.encode(img)
    emb2 = aemulti.encode(img2)
    
    decodes = []
    for lamb in range(1,10):
        lamb/=10
        embres = lamb*emb+(1-lamb)*emb2
        decode = aemulti.decode(embres).view(28,28).to(device='cpu')
        decodes.append(decode)
    
    decodes = torch.stack(decodes)
    decodes = decodes.unsqueeze(1).repeat(1,3,1,1).double()
    decodes = make_grid(decodes)
    writer.add_image(f'samples', decodes, 0)

############################################################################################
############################################################################################
############################################################################################

############################################################################################
############################ HighWay #######################################################
############################################################################################

class Highway(nn.Module):
    def __init__(self,input_dim,num_layers=10):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.H = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
        self.T = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers)])
    
    def forward(self,x):
        x_ = x

        for layer in range(self.num_layers):
            
            # --- Calcul intermédiaire des H et T
            h_out = torch.sigmoid( self.H[layer](x_) )
            t_out = torch.sigmoid( self.T[layer](x_) )
            
            # --- Mise à jour de x_
            x_ = h_out * t_out + (1 - t_out) * x_
        
        return x_

class HighwayAE(nn.Module):
    def __init__(self,input_dim,dim_out,depth):
        super().__init__()
        self.HW1 = Highway(dim_out,depth)
        self.L1 = nn.Linear(input_dim,dim_out)
        
        self.HW1t = Highway(dim_out,depth)
        self.L1t = nn.Linear(dim_out,input_dim)
        

    def forward(self,x):
        return self.decode(self.encode(x))

    def encode(self,x):
        tmp = torch.sigmoid(self.L1(x))
        tmp = torch.sigmoid(self.HW1(tmp))
        return tmp

    def decode(self,x):
        tmp = torch.sigmoid(self.HW1t(x))
        tmp = torch.sigmoid(self.L1t(tmp))

        return tmp
        

hw = HighwayAE(len(monDs[0][0]),30,3)
hw = utils.optim(hw,data_train,torch.nn.MSELoss(reduction="mean"),epoch=1000,alpha=1,nameState='hw30d3',affFunc = lambda s,lf : utils.aff_complet(s,lf,monDs,dsTest,name='hw_dim30'))

aff_exemples(hw,'hw30d3')

with torch.no_grad():
    img,_ = monDs[0]
    plt.imshow(img.view(28,28))
    plt.show()
    imghat = hw(img.to(device='cuda'))
    plt.imshow(imghat.view(28,28).to(device='cpu'))
    plt.show()

############################################################################################
############################################################################################
############################################################################################

