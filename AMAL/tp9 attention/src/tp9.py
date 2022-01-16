import logging
#from os import _EnvironCodeFunc
import re
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from MLP import MLP
from tp7.student_tp7.src.tp7 import BATCHSIZE
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir runs/tp9
import os
from torch.distributions import Categorical
from math import log

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding='utf8')), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    words.append("__PAD__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size),np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

################
BATCHSIZE = 64
alpha = 0.001
epoch = 10
device = "cuda"
emb_size = 50
################

w2i, embedding, dsTrain, dsTest = get_imdb_data(emb_size)

i2w = { v:k for k,v in w2i.items()}


embedding=nn.Embedding.from_pretrained(torch.tensor(embedding))

def collate(batch): #fonction pour dataLoader comment faire un batch
    data = [torch.LongTensor(b[0]) for b in batch]
    lens = torch.LongTensor([len(b[0]) for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    return pad_sequence(data,batch_first=True,padding_value=w2i['__PAD__']), lens, labels

dlTrain  = torch.utils.data.DataLoader(dsTrain, batch_size= BATCHSIZE, shuffle=True,collate_fn = collate)
dlTest = torch.utils.data.DataLoader(dsTest, batch_size= BATCHSIZE, collate_fn = collate)

class Module1(nn.Module):
    def __init__(self, emb, emb_size):
        super().__init__()
        self.m = MLP(emb_size,1, [])
        self.emb = emb
        
    def forward(self, x, len):
        embX = self.emb(x)
        t = torch.sum(embX, dim=1)/len.view(-1,1)
        pred = self.m(t.float())
        return pred  
    
class Module2(nn.Module):
    def __init__(self, emb, emb_size):
        super().__init__()
        self.m = MLP(emb_size,1, [])
        self.emb = emb
        self.attention = nn.Linear(emb_size, 1).float()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x, len):
        global alphaAttention
        mask = torch.where(x!= w2i['__PAD__'], 1,0).unsqueeze(2)
        embX = self.emb(x).float()
        alphaAttention = self.attention(embX)
        alphaAttention[~mask]=float('-inf')
        alphaAttention = self.soft(alphaAttention)
        t = torch.sum(embX*alphaAttention, dim=1)
        pred = self.m(t.float())
        return pred
    
class Module3(nn.Module):
    def __init__(self, emb, emb_size):
        super().__init__()
        self.m = MLP(emb_size,1, [])
        self.emb = emb
        self.linearQ = nn.Linear(emb_size, emb_size).float()
        self.linearV = nn.Linear(emb_size, emb_size).float()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x, len):
        global alphaAttention
        mask = torch.where(x!= w2i['__PAD__'], 1,0).unsqueeze(2)
        embX = self.emb(x).float()
        tchapeau = torch.sum(embX, dim=1)/len.view(-1,1)
        q = self.linearQ(tchapeau).unsqueeze(2)
        alphaAttention = embX@q
        alphaAttention[~mask]=float('-inf')
        alphaAttention = self.soft(alphaAttention)
        v = self.linearV(embX)
        t = torch.sum(v*alphaAttention, dim=1)
        pred = self.m(t.float())
        return pred
    

class Module4(nn.Module):
    def __init__(self, emb, emb_size):
        super().__init__()
        self.m = MLP(emb_size,1, [])
        self.emb = emb
        self.linearQ = nn.Linear(emb_size, emb_size).float()
        self.linearV = nn.Linear(emb_size, emb_size).float()
        self.lstm = nn.LSTM(emb_size, emb_size,1)
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x, len):
        global alphaAttention
        mask = torch.where(x!= w2i['__PAD__'], 1,0).unsqueeze(2)
        embX = self.emb(x).float()
        tchapeau = torch.sum(embX, dim=1)/len.view(-1,1)
        q = self.linearQ(tchapeau).unsqueeze(2)
        h,_ = self.lstm(embX)
        alphaAttention = h@q
        alphaAttention = self.soft(alphaAttention*mask)
        v = self.linearV(embX)
        t = torch.sum(v*alphaAttention, dim=1)
        pred = self.m(t.float())
        return pred


model = Module4(embedding, emb_size).to(device)

optim = torch.optim.Adam(params=model.parameters(),lr=alpha)

overwrite = True
nom = "testall"#.join(sorted([str(k)+"="+str(v) for k,v in params.items()]))
p = 'runs/tp9/'+nom
if os.path.isdir(p):
    if(overwrite):
        err=os.system('rmdir /S /Q "{}"'.format(p))

writer = SummaryWriter(p)

it=0
for e in range(epoch):
    l_g=0
    i=0
    for x,y,l in dlTrain:
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        predict = model(x, l)
        loss = nn.functional.binary_cross_entropy_with_logits(predict.flatten(),y.float())
        l_g += loss
        i+=1
        loss.backward()
        
        optim.step()
        if(it%200==0):
            entropy = Categorical(probs = alphaAttention.squeeze(2)).entropy()
            writer.add_histogram("Entropy", entropy, it)
        it +=1
    
    #acc 
    acc = 0
    n = 0
    for x,l,y in dlTrain:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        predict = torch.where(torch.sigmoid(model(x, l))>0.5,1,0)
        acc += (predict.flatten() == y).int().sum()
        n+=x.size(0)
    
    acc_train = (acc/n).item()
    acc = 0
    n = 0
    for x,l,y in dlTest:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        predict = torch.where(torch.sigmoid(model(x, l))>0.5,1,0)
        acc += (predict.flatten() == y).int().sum()
        n+=x.size(0)

    acc_test = (acc/n).item()
    l_g = (l_g/i).item()
    writer.add_scalar("test/acc", acc_test, e)
    writer.add_scalar("train/acc", acc_train, e)
    writer.add_scalar("train/loss", l_g,e)
    print(e,l_g,acc_train,acc_test)