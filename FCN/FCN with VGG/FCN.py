import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
"""
background
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
"""
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

train = torchvision.datasets.VOCSegmentation(root='VOC-2012/',
                                             year='2012', image_set='train', transform=transform,
                                             target_transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=4, shuffle=True)

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
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        # 8
        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.layer_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        # 4
        self.layer_5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                               stride=1, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2)
                                     )
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 21, kernel_size=1)

    def forward(self, x):
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)
        x4 = self.layer_4(x3)
        x5 = self.layer_5(x4)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def own_MSE(output, target):
    loss = torch.mean((output - target)**2)

    return loss

model = VGG()
model = model.cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
cost = 0
iterations = []
train_losses = []
print("starting")
for epoch in range(10):
    model.train()
    for X, Y in train_loader:
        torch.cuda.empty_cache()
        target = np.zeros((4, 21, 416, 416))
        for i in range(4):
            for j in range(21):
                for k in range(416):
                    for l in range(416):
                        if(Y[i][0][k][l] == j):
                            target[i][j][k][l] = j
        target = torch.tensor(target)
        X = X.to(cuda)
        target = target.to(cuda)
        optimizer.zero_grad()
        output = model(X)
        cost = own_MSE(output, target.long())
        cost.backward()
        optimizer.step()
        scheduler.step()

    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    print("lr : {:>6}".format(scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
    iterations.append(epoch)
    train_losses.append(cost.tolist())

torch.save(model.state_dict(), './vggfcn model/model.pt')

plt.subplot(111)
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.title('loss')
plt.show()
# del train_loader
# torch.cuda.empty_cache()
"""
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
"""