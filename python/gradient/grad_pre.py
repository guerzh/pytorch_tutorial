import  numpy as np
import torch
from torch.autograd import Variable



def f(x):
    return x**2 + 2 * x


x = Variable(torch.from_numpy(np.array([2.0])), requires_grad = True)





    