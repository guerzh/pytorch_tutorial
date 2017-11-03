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
y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)





loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    print(t, loss.data[0])
    
    model.zero_grad()
    loss.backward()
    optimizer.step()
    



########################################################################################
    
x = Variable(torch.from_numpy(test_x)).type(dtype_float)
np.argmax(model(x).data.numpy(), 1)
im_idx = 0
plt.imshow(test_x[im_idx, :].reshape((32, 32)), cmap = plt.cm.gray)
plt.show()
#########################################################################################

W = model[0].weight.data.numpy()
plt.imshow(W[2, :].reshape((32, 32)), cmap = plt.cm.coolwarm)
plt.show()

#########################################################################################





