import sys
sys.path.insert(0,'/tp4_RNN/student_tp4/src')

from tp4_RNN.student_tp4.src.utils import RNN,SampleMetroDataset
from tp4_RNN.student_tp4.src import tools
import torch
from torch.utils.data import DataLoader, dataloader
from pathlib import Path
from tp4_RNN.student_tp4.src.tools import State
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs

# Nombre de stations utilisé
CLASSES = 20
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32
#dim h rnn
dimH=50

PATH = "tp4_RNN/data/"

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False,drop_last=True)
data_train_l = DataLoader(ds_train,batch_size=len(ds_train),shuffle=True,drop_last=True)
data_test_l = DataLoader(ds_test, batch_size=len(ds_test),shuffle=False,drop_last=True)

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

class RNN_Classifier(torch.nn.Module):
    def __init__(self,dim_h,dim_in,dim_out):
        super().__init__()
        self.rnn = RNN(dim_h,dim_in,dim_out)

    def forward(self,x):
        hs = self.rnn(x)
        last_h = hs[:,-1,:]
        return self.rnn.decode(last_h)


clf = RNN_Classifier(dimH, DIM_INPUT, CLASSES)
loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

def aff_func(s,lf):
    writer = SummaryWriter("runs/tp4_clf_s20_l20_dim50")
    l_train = tools.compute_loss(s.model,data_train_l,loss_func)
    l_test = tools.compute_loss(s.model,data_test_l,loss_func)
    acc_train = tools.accuracy(s.model,data_train_l)
    acc_test = tools.accuracy(s.model,data_test_l)
    writer.add_scalar(f'loss/train', l_train, s.epoch)
    writer.add_scalar(f'loss/test', l_test, s.epoch)
    writer.add_scalar(f'accuracy/train', acc_train, s.epoch)
    writer.add_scalar(f'accuracy/test', acc_test, s.epoch)
    print(s.epoch,l_train,l_test,acc_train,acc_test)

def aff_iter(s,lf):
    if (s.epoch%20==0):
        print('epoch :',s.epoch)

clf = tools.optim(clf,data_train,loss_func,optmizer=torch.optim.Adam,epoch=100,alpha=0.001,nameState='rnnclf_final_s20_l20_dim50',affFunc=aff_func)

print('acc_train', tools.accuracy(clf,data_train_l))
print('acc_test', tools.accuracy(clf,data_test_l))