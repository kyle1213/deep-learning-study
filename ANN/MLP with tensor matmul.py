import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import math

random_seed = 42

torch.manual_seed(random_seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

np.random.seed(random_seed)

random.seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.cuda.manual_seed_all(random_seed)  # multi-GPU

train = torchvision.datasets.MNIST('/MNIST', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))
test = torchvision.datasets.MNIST('/MNIST', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))]))
print("hi")

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)

cuda = torch.device('cuda')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.W1 = torch.nn.Parameter(torch.empty(784, 256).cuda())
        self.W2 = torch.nn.Parameter(torch.empty(256, 128).cuda())
        self.W3 = torch.nn.Parameter(torch.empty(128, 10).cuda())

        torch.nn.init.uniform_(self.W1, a=-math.sqrt(1/784), b=math.sqrt(1/784))
        torch.nn.init.uniform_(self.W2, a=-math.sqrt(1 / 256), b=math.sqrt(1 / 256))
        torch.nn.init.uniform_(self.W3, a=-math.sqrt(1 / 128), b=math.sqrt(1 / 128))

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = torch.matmul(x, self.W1)
        x = self.relu(x)
        x = torch.matmul(x, self.W2)
        x = self.relu(x)
        x = torch.matmul(x, self.W3)
        return x


model = MLP()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
cost = 0

iterations = []
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for epoch in range(10):
    model.train()
    correct = 0
    for X, Y in train_loader:
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        hypo = model(X)
        cost = loss(hypo, Y)
        cost.backward()
        optimizer.step()
        prediction = hypo.data.max(1)[1]
        correct += prediction.eq(Y.data).sum()

    model.eval()
    correct2 = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model(data)
        cost2 = loss(output, target)
        prediction = output.data.max(1)[1]
        correct2 += prediction.eq(target.data).sum()

    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    iterations.append(epoch)
    train_losses.append(cost.tolist())
    test_losses.append(cost2.tolist())
    train_acc.append((100 * correct / len(train_loader.dataset)).tolist())
    test_acc.append((100 * correct2 / len(test_loader.dataset)).tolist())

# del train_loader
# torch.cuda.empty_cache()

model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(cuda)
    target = target.to(cuda)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

plt.subplot(121)
plt.plot(range(1, len(iterations) + 1), train_losses, 'b--')
plt.plot(range(1, len(iterations) + 1), test_losses, 'r--')
plt.subplot(122)
plt.plot(range(1, len(iterations) + 1), train_acc, 'b-')
plt.plot(range(1, len(iterations) + 1), test_acc, 'r-')
plt.title('loss and accuracy')
plt.show()
