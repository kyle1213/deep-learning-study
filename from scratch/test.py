import numpy as np
import matplotlib.pyplot as plt
import torch

x = [[1, 2], [3, 4], [5, 6]]
y = [[1, 4], [9, 16], [25, 36]]

cuda = torch.device('cuda')

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.plane_weights = torch.nn.Parameter(torch.randn(3, 2).cuda())

    def forward(self, x):
        x = self.plane_weights * x

        return x


model = Test()
model = model.cuda()

#loss = torch.nn.CrossEntropyLoss()
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
cost = 0

x = torch.tensor(x)
y = torch.tensor(y)

for epoch in range(150):
    model.train()
    torch.cuda.empty_cache()
    X = x.to(cuda)
    Y = y.to(cuda)
    optimizer.zero_grad()
    output = model(X)
    cost = loss(output, Y.float())
    cost.backward()
    optimizer.step()

for parameter in model.parameters():
    print(parameter)