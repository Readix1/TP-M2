import sys
sys.path.insert(0,'/tp4_RNN/student_tp4/src')

from tp4_RNN.student_tp4.src.utils import RNN, device,ForecastMetroDataset
from tp4_RNN.student_tp4.src import tools
import torch
from torch.utils.data import DataLoader, dataloader
from pathlib import Path
from tp4_RNN.student_tp4.src.tools import State
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 5
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32
#dim h rnn
dimH=20

PATH = "tp4_RNN/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
data_train_l = DataLoader(ds_train,batch_size=len(ds_train),shuffle=True,drop_last=True)
data_test_l = DataLoader(ds_test, batch_size=len(ds_test),shuffle=False,drop_last=True)

#  TODO:  Question 3 : Prédiction de séries temporelles

class RNN_Gene(torch.nn.Module):
    def __init__(self,dim_h,dim_in,dim_out):
        super().__init__()
        self.rnn = RNN(dim_h,dim_in,dim_out)

    def forward(self,x):
        hs = self.rnn(x)
        return self.rnn.decode(hs)

#### OPTIM #####
rnn = RNN_Gene(dimH, DIM_INPUT*CLASSES, DIM_INPUT*CLASSES)#??

loss_func = torch.nn.MSELoss(reduction="mean")
def aff_func(s,lf):
    writer = SummaryWriter("runs/tp4_gene_s10_l5_dim20")
    l_train = tools.compute_loss(s.model,data_train_l,loss_func)
    l_test = tools.compute_loss(s.model,data_test_l,loss_func)
    writer.add_scalar(f'loss/train', l_train, s.epoch)
    writer.add_scalar(f'loss/test', l_test, s.epoch)
    print(s.epoch,l_train,l_test)

rnn = tools.optim(rnn,data_train,loss_func,optmizer=torch.optim.Adam,epoch=100,alpha=0.01,nameState='rnngene_l5_dim20',affFunc=aff_func)

somme_err = 0
somme = 0
n=0
for x,y in data_train:
    x = x.cuda()
    y = y.cuda()
    pred = rnn(x)
    somme_err += torch.abs(pred-y).sum()
    somme += torch.sum(y)
    n_add = 1
    for dim in y.size():
        n_add *= dim
    n += n_add
print(somme_err/n, somme/n)