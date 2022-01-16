import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()


ctxL = Context()
ctxM = Context()

for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    loss = MSE.forward(ctxM,Linear.forward(ctxL,x,w,b),y)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    gradY,_ = MSE.backward(ctxM, 1)

    grad_x, grad_w, grad_b = Linear.backward(ctxL, gradY)

    ##  TODO:  Mise à jour des paramètres du modèle
    w = w-grad_w*epsilon
    b = b-epsilon * grad_b