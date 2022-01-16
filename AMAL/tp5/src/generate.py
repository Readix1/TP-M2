import sys
sys.path.insert(0,'/tp5/src')

from tp5.src.textloader import  string2code, id2lettre
import math
import torch
import torch.nn.functional as F
#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    res = start
    with torch.no_grad():
        embstart = emb(string2code(start)).view(len(start),1,-1)
        h = rnn.forward(embstart)[-1]
        dec = decoder(h)
        car = torch.argmax(dec).item()
        res += id2lettre[car]
        i = len(start)
        while(car != eos and i<=maxlen):
            embcar = emb(torch.tensor([car]))
            h = rnn.one_step(embcar,h)
            dec = decoder(h)
            car = torch.argmax(dec).item()
            res += id2lettre[car]
            i += 1

    return res

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200,pnuc=True):
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
    list = [(start,0)]
    with torch.no_grad():
        i=0
        while i<len(list):
            i=0
            listTemp = []
            for j,p in list:
                if j[-5:] == id2lettre[eos] or len(j)>=maxlen:
                    i+=1
                    listTemp.append((j,p))
                    continue
                embstart = emb(string2code(j)).view(len(j),1,-1)
                h = rnn.forward(embstart)[-1]
                if pnuc :
                    dec = torch.log(compute(h))
                else :
                    dec = torch.log(F.softmax(decoder(h)[0],dim=0))
                    
                car = torch.argsort(dec, descending=True)[:k]
                for l in car:
                    listTemp.append((j+id2lettre[l.item()],p+dec[l.item()].item()))
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
        dec = F.softmax(decoder(h)[0],dim=0)
        argsortdec = torch.argsort(dec, descending=True)
        dec_cumsum = torch.cumsum(dec[argsortdec],dim=0)
        i_sup_alpha = torch.argmax((dec_cumsum>=alpha).float())
        dec[argsortdec[i_sup_alpha+1:]]=0.
        dec /= dec.sum()
        return dec

    return compute
