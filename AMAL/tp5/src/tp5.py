import sys
sys.path.insert(0,'/tp5/src')
from tp5.src import tools

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tp5.src.textloader import *
from tp5.src.generate import *
import torch.nn.functional as F
import string
import unicodedata
from pathlib import Path

#  TODO: 

cle = CrossEntropyLoss(reduction='none')

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    mask = (target!=padcar).contiguous().view(-1)
    output = output.view(-1,output.size(2))
    target = target.contiguous().view(-1).long()

    return (cle(output, target)*mask).sum()/mask.sum()
    

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self,dimH, dimIn, dimOut):
        super(RNN, self).__init__()
        self.dimH = dimH
        self.dimIn=dimIn
        self.linearCache = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearDecode = nn.Linear(self.dimH, dimOut)

    def forward(self, x, h=None):
        """
            x: T * Batch * d
        """
        # print(x.size())
        device = x.device
        res = torch.empty((x.size(0),x.size(1),self.dimH),device=device)
        if h==None:
            h = torch.zeros(x.size(1), self.dimH)
            h = h.to(device)
        for i in range(x.size(0)):
            hprim = self.one_step(x[i],h)
            res[i] = hprim#torch.cat((res, hprim),1)
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


class LSTM(nn.Module):
    def __init__(self,dimH, dimIn, dimOut):
        super(LSTM, self).__init__()
        self.dimH = dimH
        self.dimIn=dimIn
        self.linearF = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearI = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearC = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearO = nn.Linear(dimIn+self.dimH,self.dimH)
        self.linearDecode = nn.Linear(self.dimH, dimOut)

    def forward(self, x, h=None):
        """
            x: T * Batch * d
        """
        # print(x.size())
        device = x.device
        res = torch.empty((x.size(0),x.size(1),self.dimH),device=device)
        if h==None:
            h = torch.zeros(x.size(1), self.dimH)
            h = h.to(device)
        self.c = torch.zeros(x.size(1), self.dimH)
        self.c = self.c.to(device)
        for i in range(x.size(0)):
            hprim = self.one_step(x[i],h)
            res[i] = hprim#torch.cat((res, hprim),1)
            h = hprim
        return res

    def decode(self, h):
        return self.linearDecode(h)

    def one_step(self, x, h):
        """
            x: Batch * d
        """
        xh = torch.cat((x,h),1)
        ft = torch.sigmoid(self.linearF(xh))
        it = torch.sigmoid(self.linearI(xh))
        self.c = ft*self.c + it * torch.tanh(self.linearC(xh)) 
        ot = torch.sigmoid(self.linearO(xh))
        hprim = ot * torch.tanh(self.c)
        return hprim


class GRU(nn.Module):
    def __init__(self,dimH, dimIn, dimOut):
        super(GRU, self).__init__()
        self.dimH = dimH
        self.dimIn=dimIn
        self.linear = nn.Linear(dimIn+self.dimH,self.dimH,bias=False)
        self.linearZ = nn.Linear(dimIn+self.dimH,self.dimH,bias=False)
        self.linearR = nn.Linear(dimIn+self.dimH,dimIn+self.dimH,bias=False)
        self.linearDecode = nn.Linear(self.dimH, dimOut)

    def forward(self, x, h=None):
        """
            x: T * Batch * d
        """
        # print(x.size())
        device = x.device
        res = torch.empty((x.size(0),x.size(1),self.dimH),device=device)
        if h==None:
            h = torch.zeros(x.size(1), self.dimH)
            h = h.to(device)
        for i in range(x.size(0)):
            hprim = self.one_step(x[i],h)
            res[i] = hprim#torch.cat((res, hprim),1)
            h = hprim
        return res

    def decode(self, h):
        return self.linearDecode(h)

    def one_step(self, x, h):
        """
            x: Batch * d
        """
        xh = torch.cat((x,h),1)
        zt = torch.sigmoid(self.linearZ(xh))
        rt = torch.sigmoid(self.linearR(xh))
        hprim = torch.tanh(self.linear(rt*xh))
        return (1-zt) * h + zt * hprim


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot


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


class Gen(nn.Module):
    def __init__(self,rnn,enc_in,enc_out,dim_h,dim_in,dim_out):
        super(Gen,self).__init__()
        self.enc = nn.Embedding(enc_in,enc_out)
        self.rnn = rnn(dim_h,dim_in,dim_out)

    def forward(self,x):
        x_enc = self.enc(x.long())
        h = self.rnn(x_enc)
        #h = torch.stack(h,dim=1)
        y = self.rnn.decode(h)
        return y


BATCH_SIZE = 32
LENGTH = 30
## Dictionnaire index -> lettre
#id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))



## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1


## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '

## Dictionnaire index -> lettre
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'

## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

PATH = "tp5/data/"
with open(PATH+'trump_full_speech.txt') as f:
    text = f.read()

ds_train = TextDataset(text,10000,LENGTH)
data_train_l = DataLoader(ds_train,batch_size=len(ds_train),drop_last=True, collate_fn = pad_collate_fn)
len_emb = len(id2lettre)


DIM_EMB = 50
DIM_H = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Gen(GRU,len_emb,DIM_EMB,DIM_H,DIM_EMB,len_emb)

lossfunc = lambda yhat,y : maskedCrossEntropy(yhat,y,PAD_IX)
alpha=0.005
epoch=200
nameState='testgru_100'
dataLoader = data_train_l
optmizer=torch.optim.Adam

#affichage
def affichage(state,lossfunc):
    with torch.no_grad():
        writer = SummaryWriter("runs/tp5_gru500")
        x,y = next(iter(data_train_l))
        x = x.to(device)
        y = y.to(device)
        predict = state.model(x)
        l_train = lossfunc(predict,y).item()

        #mask pour accuracy
        mask = (y!=PAD_IX)

        acc = ((torch.where(torch.argmax(predict,axis=2)==y,1.,0.)*mask).sum()/mask.sum()).item()
        
        print('epoch',state.epoch,'loss train',l_train,'acc',acc)
        writer.add_scalar(f'loss/train', l_train, state.epoch)
        writer.add_scalar(f'accuracy/train', acc, state.epoch)

model = tools.optim(model,dataLoader,lossfunc,optmizer,epoch,alpha,nameState,affichage)

model = model.to('cpu')
start = 'You '
print('generate :',generate(model.rnn,model.enc,model.rnn.decode,EOS_IX,start,100))
print('generate_beam sans nuclear sampling k=3:',generate_beam(model.rnn,model.enc,model.rnn.decode,EOS_IX,3,start,100,pnuc=False))
print('generate_beam avec nuclear sampling k=3:',generate_beam(model.rnn,model.enc,model.rnn.decode,EOS_IX,3,start,100,pnuc=True))
print('generate_beam sans nuclear sampling k=10:',generate_beam(model.rnn,model.enc,model.rnn.decode,EOS_IX,10,start,100,pnuc=False))
print('generate_beam avec nuclear sampling k=10:',generate_beam(model.rnn,model.enc,model.rnn.decode,EOS_IX,10,start,100,pnuc=True))