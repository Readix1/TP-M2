import sys

from torch.nn.functional import cross_entropy
sys.path.insert(0,'/tp6/src')
from tp6.src import tools

import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)




#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

class posTagging(nn.Module):
    def __init__(self, enc_in, enc_out, hsize, out) -> None:
        super().__init__()
        self.enc = nn.Embedding(enc_in,enc_out)
        self.lstm = nn.LSTM(enc_out, hsize,bidirectional=True)#, proj_size = enc_in)
        self.linear = nn.Linear(hsize*2, out)

    def forward(self, x):
        x_enc = self.enc(x)#.long()
        h,_ = self.lstm(x_enc)
        res = self.linear(h)
        return res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def affichage(state,lossfunc):
    writer = SummaryWriter("runs/tp6_pos_lstmbi200")
    with torch.no_grad():
        l_train = 0
        n_acc = 0
        n= 0
        acc_train = 0
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            predict = state.model(x)
            l_train += lossfunc(predict,y).item()
            
            mask = (y!=0)
            acc_train += (torch.where(torch.argmax(predict,axis=2)==y,1.,0.)*mask).sum().item()
            n_acc += mask.sum().item()
            n+=1
        
        l_train /= n    
        acc_train/=n_acc

        l_test = 0
        n_acc = 0
        n = 0
        acc_test = 0
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            predict = state.model(x)
            l_test += lossfunc(predict,y).item()
            
            mask = (y!=0)
            acc_test += (torch.where(torch.argmax(predict,axis=2)==y,1.,0.)*mask).sum().item()
            n_acc += mask.sum().item()
            n+=1
            
        l_test /= n
        acc_test/=n_acc
        
        print('epoch',state.epoch,'loss train',l_train,'acc train',acc_train,'loss test',l_test,'acc test',acc_test)
        writer.add_scalar(f'loss/train', l_train, state.epoch)
        writer.add_scalar(f'loss/test', l_test, state.epoch)
        writer.add_scalar(f'accuracy/train', acc_train, state.epoch)
        writer.add_scalar(f'accuracy/test', acc_test, state.epoch)

dim_h = 200
model = posTagging(len(words), 100, dim_h, len(tags))

loss = CrossEntropyLoss(ignore_index=0)
lossfunc = lambda x,y : loss(torch.flatten(x,0,1),torch.flatten(y))
optim = torch.optim.Adam

model = tools.optim(model, train_loader, lossfunc,optim, 10, 0.001, 'tp6_pos_lstmbi_200', affichage)

model = model.cpu()
for i in torch.randint(len(test_data),(10,)):
    x,y = test_data[i]
    phrase = " ".join(words.getwords(x))
    res = " ".join(tags.getwords(y))
    x = torch.Tensor(x).long()
    y = torch.Tensor(y).long()
    pred = torch.argmax(model(x.unsqueeze(0)),dim=2)[0]
    pred = " ".join(tags.getwords(pred))

    print("phrase :", phrase)
    print("tag :",res)
    print('predict :',pred)
    print()