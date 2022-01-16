import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime

def compute_loss(model, data, ydata,lossfunc):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    ydata = ydata.to(device)

    with torch.no_grad():
        predict = model(data)
        loss = lossfunc(predict,ydata)

    return loss

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
def print_loss(state,lossfunc,dstrain,dstest):
    x,y = dstrain[:]
    train_loss = compute_loss(state.model,x,y,lossfunc)
    x,y = dstest[:]
    test_loss = compute_loss(state.model,x,y,lossfunc)
    print("nb epoch:",state.epoch, "| nb iter:", state.iter, "| Train_loss:",train_loss.item(), "| Testing_loss:", test_loss.item() )

def tensorboard_aff(state,lossfunc,dstrain,dstest,name='runs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    writer = SummaryWriter("runs/"+name)
    x,y = dstrain[:]
    train_loss = compute_loss(state.model,x,y,lossfunc)
    x,y = dstest[:]
    test_loss = compute_loss(state.model,x,y,lossfunc)
    writer.add_scalar(f'train/loss', train_loss, state.epoch)
    writer.add_scalar(f'test/loss', test_loss, state.epoch)

def aff_complet(state,lossfunc,dstrain,dstest,name='runs_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    writer = SummaryWriter("runs/"+name)
    x,y = dstrain[:]
    train_loss = compute_loss(state.model,x,y,lossfunc)
    x,y = dstest[:]
    test_loss = compute_loss(state.model,x,y,lossfunc)
    print("nb epoch:",state.epoch, "| nb iter:", state.iter, "| Train_loss:",train_loss.item(), "| Testing_loss:", test_loss.item() )
    writer.add_scalar(f'train/loss', train_loss, state.epoch)
    writer.add_scalar(f'test/loss', test_loss, state.epoch)

#### optim ####
def optim(model, dataLoader,lossfunc,optmizer=torch.optim.SGD, epoch = 10, alpha = 1,momentum=0, nameState='model',affFunc=None):

    #permet de selectionner le gpu si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    savepath = Path(nameState+".pch")
    
    if savepath.is_file():
        state = State.load(savepath)
        state.model = state.model.to(device)
        state.optim = optim = optmizer(params=state.model.parameters(),momentum=momentum,lr=alpha) ## on optimise selon w et b, lr : pas de gradient
    else:
        model = model.to(device)
        optim = optmizer(params=model.parameters(),momentum=momentum,lr=alpha) ## on optimise selon w et b, lr : pas de gradient
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