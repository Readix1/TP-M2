import torch
from pathlib import Path
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs

def compute_loss(model, ds,lossfunc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x,y = next(iter(ds))
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        predict = model(x)
        loss = lossfunc(predict,y)

    return loss.item()

def compute_train_test_loss(model,dstrain,dstest,lossfunc):
    train_loss = compute_loss(model,dstrain,lossfunc)
    test_loss = compute_loss(model,dstest,lossfunc)
    return train_loss,test_loss


class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iter = 0
    
    def save(self,path):
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(path):
        with path.open("rb") as fp:
            state = torch.load(fp) #on recommence depuis le modele sauvegarde
            return state

##### fonction d'affichage pendant l'optim #####
def print_loss(state,dstrain,dstest,lossfunc):
    train_loss, test_loss = compute_train_test_loss(state.model,dstrain,dstest,lossfunc)
    print("nb epoch:",state.epoch, "| nb iter:", state.iter, "| Train_loss:",train_loss, "| Testing_loss:", test_loss)

def print_acc(state,dstrain,dstest):
    accuracy_train = accuracy(state.model,dstrain)
    accuracy_test = accuracy(state.model,dstest)
    print("nb epoch:",state.epoch, "| nb iter:", state.iter, "| acc_train:",accuracy_train, "| acc_loss:", accuracy_test)

def tensorboard_aff(state,lossfunc,train_dataloader,test_dataloader,name='runs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    writer = SummaryWriter("runs/"+name)

    train_loss, test_loss = compute_train_test_loss(state.model,train_dataloader,test_dataloader,lossfunc)
    
    writer.add_scalar(f'train/loss', train_loss, state.epoch)
    writer.add_scalar(f'test/loss', test_loss, state.epoch)

def aff_complet(state,lossfunc,train_dataloader,test_dataloader,name='runs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    writer = SummaryWriter("runs/"+name)
    
    train_loss, test_loss = compute_train_test_loss(state.model,train_dataloader,test_dataloader,lossfunc)
    
    print("nb epoch:",state.epoch, "| nb iter:", state.iter, "| Train_loss:",train_loss, "| Testing_loss:", test_loss )
    writer.add_scalar(f'train/loss', train_loss, state.epoch)
    writer.add_scalar(f'test/loss', test_loss, state.epoch)

#### optim ####
def optim(model, dataLoader,lossfunc,optmizer=torch.optim.SGD,epoch = 10, alpha = 1, nameState='model',affFunc=None):

    #permet de selectionner le gpu si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    savepath = Path(nameState+".pch")
    
    if savepath.is_file():
        state = State.load(savepath)
        state.model = state.model.to(device)
        state.optim = optim = optmizer(params=state.model.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    else:
        model = model.to(device)
        optim = optmizer(params=model.parameters(),lr=alpha) ## on optimise selon w et b, lr : pas de gradient
        state = State(model, optim)

    for epoch in range (state.epoch, epoch):
        if affFunc:
            affFunc(state,lossfunc)
        
        for x,y in dataLoader:
            state.optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            predict = state.model(x)
            l = lossfunc(predict,y)
            l.backward()

            state.optim.step()
            state.iter += 1

        with savepath.open("wb") as fp:
            state.epoch = epoch + 1
            torch.save(state, fp)

    if affFunc:
        affFunc(state,lossfunc)
    
    return state.model

def accuracy(model,data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x,y = next(iter(data))
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    predict = F.softmax(model(x),dim=1)
    return torch.mean(torch.where(torch.argmax(predict,axis=1)==y,1.,0.)).item()