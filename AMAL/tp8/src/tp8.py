import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp8.src.tp8_preprocess import TextDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import time

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration


########################## HYPERPARAMETRE #####################
# Taille du vocabulaire
vocab_size = 1000
emb_size = 50
MAINDIR = Path("tp8\src")

LOG_PATH = "runs\tp8"

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{MAINDIR}\{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


class firstCNN(pl.LightningModule):
    def __init__(self,filters=[],kernel_size=3,stride=1,emb_size=emb_size,voc_size=vocab_size,learning_rate=1e-3):
        super().__init__()
        self.emb = nn.Embedding(voc_size, emb_size)
        seq= []
        dimprec=emb_size
        for fil in filters:
            seq.append(nn.Conv1d(dimprec, fil, kernel_size, stride))
            seq.append(nn.MaxPool1d(kernel_size, kernel_size))
            seq.append(nn.ReLU())
            dimprec = fil
        self.model = nn.Sequential(*seq)
        self.lin = nn.Linear(fil, 3)
        
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "firstCNN lightning"

    def forward(self,x):
        """ Définit le comportement forward du module"""
        x = self.forwardCONV(x)
        x = torch.max(x, dim=2)[0]
        x = self.lin(x)
        return x
    
    def forwardCONV(self,x):
        """ Définit le comportement forward du module"""
        x = torch.transpose(self.emb(x), 1, 2).float()
        x = self.model(x)
        return x
    

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        yhat= self(x) ## equivalent à self.model(x)
        loss = self.loss(yhat,y)
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("accuracy",acc/len(x),on_step=False,on_epoch=True)
        return logs
    
    def training_step(self,batch,batch_idx):
        return self.step(batch,batch_idx)
    
    def test_step(self,batch,batch_idx):
        return self.step(batch,batch_idx)
    
    def validation_step(self,batch,batch_idx):
        return self.step(batch,batch_idx)
    
model = firstCNN([32, 32])

logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime().replace(':', '-'),default_hp_metric=False)

    
trainer = pl.Trainer(default_root_dir=LOG_PATH,logger=logger,max_epochs=100)
trainer.fit(model,train_iter)
trainer.test(model,test_iter)