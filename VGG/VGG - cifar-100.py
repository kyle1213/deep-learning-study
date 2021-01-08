import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR100(root='CIFAR-100/',
                                      train=True, transform=transform,
                                      download=True)
test = torchvision.datasets.CIFAR100(root='CIFAR-100/',
                                     train=False, transform=transform,
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=512, shuffle=True)
train_test_loader = torch.utils.data.DataLoader(dataset=train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=128, shuffle=True)

cuda = torch.device('cuda')


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # 16
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                                             stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                             stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        # 8
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        # 4
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer_5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer_6 = nn.Sequential(nn.Flatten(),
                                     nn.Linear(4*4*512, 4096, bias=True),
                                     nn.Linear(4096, 4096, bias=True),
                                     nn.Linear(4096, 100, bias=True)
                                     )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        x = self.layer_6(x)
        return x


model = VGG()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

cost = 0

for epoch in range(70):
    for X, Y in train_loader:
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        hypo = model(X)
        cost = loss(hypo, Y)
        cost.backward()
        optimizer.step()

    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))


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

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset))
      )
correct = 0
for data, target in train_test_loader:
    data = data.to(cuda)
    target = target.to(cuda)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Train set: Accuracy: {:.2f}%'.format(100. * correct / len(train_test_loader.dataset)))











