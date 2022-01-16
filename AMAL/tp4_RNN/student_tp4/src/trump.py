import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from pathlib import Path
from tp4_RNN.student_tp4.src.utils import RNN, device
from tp4_RNN.student_tp4.src.tools import State

BATCH_SIZE = 32
LENGTH = 10
## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 
PATH = "tp4_RNN/data/"
with open(PATH+'trump_full_speech.txt') as f:
    text = f.read()

ds = TrumpDataset(text,200,LENGTH)
data_train = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
data_train_l = DataLoader(ds,batch_size=len(ds),shuffle=True,drop_last=True)
len_emb = len(id2lettre)

def onehot(x,len_emb):
    len_seq = x.size(1)
    res = torch.zeros((x.size(0),len_seq,len_emb))
    for i in range(x.size(0)):
        res[i,torch.arange(len_seq),x[i]]=1
    return res

class Encoder(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Encoder,self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in,dim_out)
        self.biais = nn.Parameter(torch.rand(dim_in))

    def forward(self,x):
        return self.encode(x)
    
    def encode(self,x):
        return self.linear(x)

    def decode(self,x):
        return F.linear(x,self.linear.weight.t(),self.biais)

DIM_EMB = 50
DIM_H = 10
device=torch.device('cpu')

class Gen(nn.Module):
    def __init__(self,enc_in,enc_out,dim_h,dim_in,dim_out,device):
        super(Gen,self).__init__()
        self.enc = Encoder(enc_in,enc_out)
        self.rnn = RNN(dim_h,dim_in,dim_out,device)

    def forward(self,x):
        x_enc = self.enc.encode(x)
        h = self.rnn(x_enc)
        #h = h.view(-1,h.size(2))#torch.stack(h,dim=1)
        y = self.rnn.decode(h)
        y_dec = self.enc.decode(y)
        return y_dec

model = Gen(len_emb,DIM_EMB,DIM_H,DIM_EMB,DIM_EMB,device)

lossfunc = nn.CrossEntropyLoss()
alpha=0.1
epoch=1000
nameState='rnn4'
dataLoader = data_train
optmizer=torch.optim.SGD

savepath = Path(nameState+".pch")
if savepath.is_file():
    state = State.load(savepath)
    state.model = state.model.to(device)
else:
    model = model.to(device)
    optim = optmizer(params=model.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    state = State(model, optim)

for epoch in range (state.epoch, epoch):
    
    state.iter = 0
    for x,y in dataLoader:
        x = onehot(x,len_emb)
        x = x.to(device)
        y = y.to(device)
        state.optim.zero_grad()
        predict = state.model(x)
        l = lossfunc(predict.transpose(1,2),y)
        l.backward()

        state.optim.step()
        state.iter += 1

    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)
    
    #affichage
    with torch.no_grad():
        x,y = next(iter(data_train_l))
        x = onehot(x,len_emb)
        x = x.to(device)
        y = y.to(device)
        predict = state.model(x)
        l_train = lossfunc(predict.transpose(1,2),y).item()
        
        print('epoch',epoch,'loss train',l_train)
