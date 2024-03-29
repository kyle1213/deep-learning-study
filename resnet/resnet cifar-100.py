import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


train_transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

train = torchvision.datasets.CIFAR100(root='D:\datasets\CIFAR-100',
                                      train=True, transform=train_transform,
                                      download=True)
test = torchvision.datasets.CIFAR100(root='D:\datasets\CIFAR-100',
                                     train=False, transform=test_transform,
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=100, shuffle=True)

cuda = torch.device('cuda')


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)

        self.batch1 = nn.BatchNorm2d(128)
        self.batch2 = nn.BatchNorm2d(256)
        self.batch3 = nn.BatchNorm2d(512)

        # 32 32 3
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 32 32 64
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # 32 32 64
        self.conv3_1_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        # 16 16 128
        self.conv3_2_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        # 16 16 128
        self.conv4_1_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        # 8 8 256
        self.conv4_2_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        # 8 8 256
        self.conv5_1_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        # 4 4 512
        self.conv5_2_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2 2 512
        self.fc_layer = nn.Sequential(
            nn.Linear(2 * 2 * 512, 100)
        )

    def forward(self, x):
        x = self.conv1_layer(x)
        shortcut = x

        x = self.conv2_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = x

        x = self.conv2_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv1(x)
        shortcut = self.batch1(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv3_1_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = x

        x = self.conv3_2_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv2(x)
        shortcut = self.batch2(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv4_1_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = x

        x = self.conv4_2_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = self.conv3(x)
        shortcut = self.batch3(shortcut)
        shortcut = nn.ReLU()(shortcut)

        x = self.conv5_1_layer(x)
        x = nn.ReLU()(x + shortcut)
        shortcut = x

        x = self.conv5_2_layer(x)
        x = nn.ReLU()(x + shortcut)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


model = ResNet()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5) # 10 epochs
cost = 0

iterations = []
train_losses = []
test_losses = []
train_acc = []
test_acc = []

for epoch in range(100):
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
        scheduler.step()
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
    train_acc.append((100*correct/len(train_loader.dataset)).tolist())
    test_acc.append((100*correct2/len(test_loader.dataset)).tolist())


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
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.plot(range(1, len(iterations)+1), test_losses, 'r--')
plt.subplot(122)
plt.plot(range(1, len(iterations)+1), train_acc, 'b-')
plt.plot(range(1, len(iterations)+1), test_acc, 'r-')
plt.title('loss and accuracy')
plt.show()