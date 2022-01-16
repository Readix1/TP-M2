import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_fn(worker_id):
    np.random.seed(12 + worker_id)

seed = torch.Generator()
seed.manual_seed(0)

class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self,dimH, dimIn, dimOut):
        super(RNN, self).__init__()
        self.dimH = dimH
        self.dimIn=dimIn
        self.linearCache = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearDecode = nn.Linear(self.dimH, dimOut)

    def forward(self, x, h=None):
        """
            x: Batch * T * d
        """
        device = x.device
        res = torch.empty((x.size(0),x.size(1),self.dimH),device=device)
        if h==None:
            h = torch.zeros(x.size(0), self.dimH)
        
        h = h.to(device)
        for i in range(x.size(1)):
            hprim = self.one_step(x[:,i,:],h)
            res[:,i,:] = hprim
            h = hprim
        return res

    def decode(self, h):
        return self.linearDecode(h)

    def one_step(self, x, h):
        """
            x: Batch * d
        """
        xh = torch.cat((x,h),1)
        hprim = torch.tanh(self.linearCache(xh))
        return hprim


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        else:
            self.stations_max = stations_max
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        if stations_max is None:
            ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
            self.stations_max = torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        else:
            self.stations_max = stations_max
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return torch.flatten(self.data[day,timeslot:(timeslot+self.length-1)],start_dim=1),torch.flatten(self.data[day,(timeslot+1):(timeslot+self.length)],start_dim=1)



# xtrain, xtest = torch.load("../../data/hzdataset.pch")

# trainds  = SampleMetroDataset(xtrain[:,:,:10,:])
# testds =  SampleMetroDataset(xtest[:,:,:10,:], stations_max = trainds.stations_max)

# data_train = DataLoader(trainds, batch_size=10, worker_init_fn = _init_fn, generator = seed)
# data_test = DataLoader( testds,  batch_size=len(xtest), worker_init_fn = _init_fn, generator = seed)




