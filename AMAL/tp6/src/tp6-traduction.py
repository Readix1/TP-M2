import logging
from torch.nn.modules.linear import Linear
from torch.nn.modules.sparse import Embedding
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader, dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
from torch.nn import CrossEntropyLoss
import mytools
import torch.nn.functional as F

import time
import re
from torch.utils.tensorboard import SummaryWriter

import datetime
from mytools import State
import random


logging.basicConfig(level=logging.INFO)

FILE = "tp6/src/data/en-fra.txt"

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device('cuda')#"cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

gen = torch.Generator()
gen.manual_seed(1234)
lines = [lines[x] for x in torch.randperm(len(lines),generator=gen)]#x% du dataset de base pour les test [:int(0.2*len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage
class TranslateEncoder(nn.Module):
    def __init__(self, enc_in, enc_out, h):
        super().__init__()
        self.emb = Embedding(enc_in, enc_out)
        self.gru = nn.GRU(enc_out, h)
        
    def forward(self,x):
        _,hs = self.gru(self.emb(x))
        return hs

class TranslateDecoder(nn.Module):
    def __init__(self, enc_in, enc_out, h):
        super().__init__()
        self.emb = Embedding(enc_in, enc_out)
        self.gru = nn.GRU(enc_out, h)
        self.linear = nn.Linear(h, enc_in)#decodeur
        self.output_size = enc_in
        
        
    def forward(self,x,h):
        embx = self.emb(x)
        hy, hs = self.gru(embx, h)
        return self.linear(hy),hs

    def generate(self,hidden,lenseq=None):
        """batch"""
        device = hidden.device
        x = torch.tensor([[vocFra.get('SOS')]*hidden.size(1)]).to(device)
        outs = torch.empty((lenseq,hidden.size(1),self.output_size)).to(device)
        i=0
        while(i<lenseq):
            out,hidden = self.forward(x,hidden)
            outs[i] = out
            
            #sample
            # prob = F.softmax(out,dim=2)
            # prob_cum = torch.cumsum(prob,dim=2)
            # rand = torch.rand(out.size(1)).to(device)
            # x = torch.argmax(torch.where(prob_cum.transpose(2,1)>rand,1,0),dim=1).detach()# detach from history as input
            
            #argmax
            x = torch.argmax(out,axis=2).detach()
            
            i+=1

        return outs

    # def generate(self,hidden,lenseq=None):
    #     device = hidden.device
    #     x = torch.tensor([[vocFra.get('SOS')]]).to(device)
    #     outs = []
    #     i=0
    #     while(i<lenseq or x.item()==vocFra.get('EOS')):
    #         out,hidden = self.forward(x,hidden)
    #         outs.append(out)
    #         #sample
    #         prob = F.softmax(out,dim=2)
    #         prob_cum = torch.cumsum(prob,dim=2)
    #         rand = torch.rand(out.size(1)).to(device)
    #         x = torch.argmax(torch.where(prob_cum.transpose(2,1)>rand,1,0),dim=1)
    #         i+=1
    #     outs = torch.cat(outs,dim=0)

    #     if(lenseq and i<lenseq):
    #         outs = torch.cat(outs,torch.zeros((lenseq-i,outs.size(1),outs.size(2)),dtype=torch.float32),dim=0)
    #         outs[i+1,:,vocFra.get('EOS')]=1.

    #     return outs

def compute_loss_acc(encoder,decoder,data):
        l_train = 0
        n = 0
        acc = 0
        n_acc = 0
        for x,lenx,y,leny in data:
            x = x.to(device)
            y = y.to(device)
            h = encoder(x)
            predict = decoder.generate(h,max(leny))
            l = lossfunc(predict,y)
            l_train += l.item()
            n+=1

            mask = (y!=0)
            choix = torch.argmax(predict,axis=2)==y
            

            acc += (torch.where(choix,1.,0.)*mask).sum().item()
            n_acc += mask.sum().item()
            
        return l_train/n,acc/n_acc
        
###################################### generate ########################################
def generate_beam(rnn, emb, decoder, eos, k,hidden, start='SOS', maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    compute = p_nucleus(decoder, 0.95)
    list = [([vocFra.get(start)],0)]
    with torch.no_grad():
        i=0
        while i<len(list):
            i=0
            listTemp = []
            for j,p in list:
                if j[-1] == eos or len(j)>=maxlen:
                    i+=1
                    listTemp.append((j,p))
                    continue
                embstart = emb(torch.tensor(j).view(-1,1)).view(len(j),1,-1)
                _,h = rnn.forward(embstart,hidden)
                #dec = torch.log(F.softmax(decoder(h),dim=-1).flatten())
                #print(compute(h))
                dec = torch.log(compute(h))
                car = torch.argsort(dec, descending=True)[:k]
                for l in car:
                    listTemp.append((j+[l.item()],p+dec[l.item()].item()))
            list = sorted(listTemp, key=lambda x : x[1], reverse= True)[:k]
            
    res = sorted(list, key=lambda x : x[1], reverse= True)[0][0]
    return res


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        dec = F.softmax(decoder(h),dim=-1).flatten()
        argsortdec = torch.argsort(dec, descending=True)
        #print('dec sort :',dec[argsortdec])
        #print('arg sort :',argsortdec)
        dec_cumsum = torch.cumsum(dec[argsortdec],dim=0)
        i_sup_alpha = torch.argmax((dec_cumsum>=alpha).float())
        dec[argsortdec[i_sup_alpha+1:]]=0.
        dec /= dec.sum()
        return dec

    return compute


############################################# OPTIM ###############################################

nameState = 'encodeRNN_full_final'
nameStateDecode = 'decodeRNN_full_final'

optmizer = torch.optim.Adam

alpha = 0.001

h_size = 1000
out_size = 500

EPOCH = 7

teacher = 0.8
decay = 0.1

savepath = Path(nameState+".pch")
savepathDecode = Path(nameStateDecode+".pch")

loss = CrossEntropyLoss(ignore_index=0)
lossfunc = lambda x,y : loss(torch.flatten(x,0,1),torch.flatten(y))

aff=mytools.LossAccAffiche(train_loader,test_loader,lossfunc,device) 

writer = SummaryWriter("runs/translate-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


if savepath.is_file():
    stateEncode = State.load(savepath)
    stateEncode.model = stateEncode.model.to(device)
    #stateEncode.optim =  optmizer(params=stateEncode.model.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    
    stateDecode = State.load(savepathDecode)
    stateDecode.model = stateDecode.model.to(device)
    #stateDecode.optim = optmizer(params=stateDecode.model.parameters(),lr=alpha)
else:
    modelEncode = TranslateEncoder(len(vocEng),out_size, h_size).to(device)
    optimEncode = optmizer(params=modelEncode.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    stateEncode = State(modelEncode, optimEncode)
    
    modelDecode = TranslateDecoder(len(vocFra),out_size, h_size).to(device)
    optimDecode = optmizer(params=modelDecode.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    stateDecode = State(modelDecode, optimDecode)

l_train=0
acc_train = 0
with torch.no_grad():
    l_train,acc_train = compute_loss_acc(stateEncode.model,stateDecode.model,train_loader)
    l_test,acc_test = compute_loss_acc(stateEncode.model,stateDecode.model,test_loader)
    print('Loss train at the epoch',stateEncode.epoch,':', l_train,'accuracy train :',acc_train,'loss test :',l_test,'accuracy test',acc_test)
    writer.add_scalar(f"train/loss",l_train,stateEncode.epoch)
    writer.add_scalar(f"test/loss",l_test,stateEncode.epoch)
    writer.add_scalar(f"train/accuracy",acc_train,stateEncode.epoch)
    writer.add_scalar(f"test/accuracy",acc_test,stateEncode.epoch)

for e in range (stateEncode.epoch, EPOCH):
    for x,lenx,y,leny in train_loader:
        stateEncode.optim.zero_grad()
        stateDecode.optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        h = stateEncode.model(x)
        
        if(random.random()<teacher):
            #ajout de sos 
            x2 = y[:-1,:]
            sos = (torch.ones((1,x2.size(1)))*vocFra.get('SOS')).long().to(device)
            x2 = torch.cat((sos,x2),dim=0)
            predict,_ = stateDecode.model(x2,h)
        else:
            #predict = [stateDecode.model.generate(h[:,ih,:].view(h.size(0),1,h.size(2)),leny[ih]) for ih in range(h.size(1))]
            #predict = pad_sequence(predict)
            predict = stateDecode.model.generate(h,max(leny))
            # on compte pas les erreurs aprés eos
            argmaxy =  torch.argmax(torch.where(torch.argmax(predict,dim=2) == 1,1,0),dim=0)
            for i,iy in enumerate(argmaxy):
                y[iy+1:,i] = 0
            #predict = predict.view((predict.size(0),predict.size(1),predict.size(3)))
        
        l = lossfunc(predict,y)
        l.backward()
        
        stateEncode.optim.step()
        stateEncode.iter += 1
        stateDecode.optim.step()
        stateDecode.iter += 1
        
        stateEncode.epoch = e + 1
        stateDecode.epoch = e + 1
        if(e == EPOCH-1):
            with savepath.open("wb") as fp:
                torch.save(stateEncode, fp)
            
            with savepathDecode.open("wb") as fp:
                torch.save(stateDecode, fp)

    teacher-=decay
    with torch.no_grad():
        l_train,acc_train = compute_loss_acc(stateEncode.model,stateDecode.model,train_loader)
        l_test,acc_test = compute_loss_acc(stateEncode.model,stateDecode.model,test_loader)
        print('Loss train at the epoch',stateEncode.epoch,':', l_train,'accuracy train :',acc_train,'loss test :',l_test,'accuracy test',acc_test)
        writer.add_scalar(f"train/loss",l_train,stateEncode.epoch)
        writer.add_scalar(f"test/loss",l_test,stateEncode.epoch)
        writer.add_scalar(f"train/accuracy",acc_train,stateEncode.epoch)
        writer.add_scalar(f"test/accuracy",acc_test,stateEncode.epoch)

def evaluate(dataset):
    with torch.no_grad():
        encoder = stateEncode.model.to('cpu')
        decoder = stateDecode.model.to('cpu')
        for i in random.choices(range(len(dataset)),k=10):
            x,y = dataset[i]

            print('x :',' '.join(vocEng.getwords(x)))
            print('y :',' '.join(vocFra.getwords(y)))
            x = x.view(-1,1)
            h = encoder(x)
            p = decoder.generate(h,lenseq=len(y))
            p = torch.argmax(p,axis=2).flatten()
            print('generate :',' '.join(vocFra.getwords(p)))
            p2 = generate_beam(decoder.gru,decoder.emb,decoder.linear,vocFra.get('EOS'),10,h)
            print('beams :',' '.join(vocFra.getwords(p2)))
            print()
