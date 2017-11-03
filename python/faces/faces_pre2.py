import numpy as np

from torch.autograd import Variable
import torch

from make_face_dataset import make_dataset

import matplotlib.pyplot as plt


act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
train_x, train_y = make_dataset(range(100), act)
test_x, test_y = make_dataset(range(100,120),act)


im_idx = 0
plt.imshow(train_x[im_idx, :].reshape((32, 32)), cmap = plt.cm.gray)
plt.show()


N_train = train_x.shape[0]

dim_x = 1024
dim_h = 20
dim_out = 6



dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
y = Variable(torch.from_numpy(train_y.astype(float)), requires_grad=False).type(dtype_float)



#def model(x, b0, W0, b1, W1):
#
#
#





logSoftMax = torch.nn.LogSoftmax() # We'll be too lazy to define this one by hand
loss = -torch.mean(torch.sum(y * logSoftMax(y_out), 1))

learning_rate = 1e-1

for t in range(1000):
    y_out = model(x, b0, W0, b1, W1)
    loss = -torch.mean(torch.sum(y * logSoftMax(y_out), 1))
    loss.backward()





x_test_all_var = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)





