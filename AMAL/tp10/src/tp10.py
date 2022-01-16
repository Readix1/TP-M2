
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from MLP import MLP
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir runs/tp10
import os
from torch.distributions import Categorical
from math import log



MAX_LENGTH = 500

logging.basicConfig(level=logging.INFO)

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
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

#  TODO: 

# @click.command()
# @click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
# @click.option('--epochs', default=50, help='Number of epochs.')
# @click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
# @click.option('--emb-size', default=100, help='embeddings size')
# @click.option('--batch-size', default=20, help='batch size')
def main(epochs=50,test_iterations=1000,modeltype=True,emb_size=100,batch_size=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.Tensor(labels).to(device), torch.Tensor(lens).to(device)


    train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False)
    return train_loader, test_loader, word2id, embeddings, emb_size


class SelfAttention(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.linearQ = nn.Linear(emb_size, emb_size)
        self.linearV = nn.Linear(emb_size, emb_size)
        self.linearK = nn.Linear(emb_size, emb_size)
        self.m = MLP(emb_size,emb_size, [])
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        k = self.linearK(x)
        v = self.linearV(x)
        q = self.linearQ(x)
        alphaAttention = torch.bmm(q, k.transpose(1,2))
        alphaAttention = self.soft(alphaAttention)
        t = torch.sum(v.unsqueeze(2)*alphaAttention.unsqueeze(3), dim=1)
        pred = torch.relu(self.m(t))
        return pred


class Module(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.sa1 = SelfAttention(emb_size)
        self.sa2 = SelfAttention(emb_size)
        self.sa3 = SelfAttention(emb_size)
        self.m = MLP(emb_size,1, [])
        
    def forward(self, x, len):
        pred = self.sa3(self.sa2(self.sa1(x)))
        return self.m(torch.sum(pred, dim = 1)/len.view(-1,1))

dlTrain, dlTest, w2i, embedding, emb_size = main()


###########################
device = 'cuda'
alpha = 0.001
epoch = 5

###########################

model = Module(emb_size).to(device)

optim = torch.optim.Adam(params=model.parameters(),lr=alpha)

overwrite = True
nom = "test"
p = 'runs/tp10/'+nom
if os.path.isdir(p):
    if(overwrite):
        err=os.system('rmdir /S /Q "{}"'.format(p))

writer = SummaryWriter(p)

it=0
for e in range(epoch):
    l_g=0
    i=0
    acc = 0
    n = 0
    for x,y,l in dlTrain:
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        predict = model(x, l)
        loss = nn.functional.binary_cross_entropy_with_logits(predict.view(-1),y)
        l_g += loss
        i+=1
        loss.backward()
        
        y_hat = torch.where(torch.sigmoid(predict)>0.5,1,0)
        acc += (y_hat.view(-1) == y).int().sum()
        n+=x.size(0)
        
        optim.step()
    
    acc_train = (acc/n).item()
    
    acc = 0
    n = 0
    for x,y,l in dlTest:
        x = x.to(device)
        y = y.to(device)
        l = l.to(device)
        predict = torch.where(torch.sigmoid(model(x, l))>0.5,1,0)
        acc += (predict.view(-1) == y).int().sum()
        n+=x.size(0)

    acc_test = (acc/n).item()
    l_g = (l_g/i).item()
    writer.add_scalar("test/acc", acc_test, e)
    writer.add_scalar("train/acc", acc_train, e)
    writer.add_scalar("train/loss", l_g,e)
    print(e,l_g,acc_train,acc_test)