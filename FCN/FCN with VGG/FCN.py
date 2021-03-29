import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary as summary_
from d2l import torch as d2l
import os


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')


def read_voc_images(_voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(_voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            _voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            _voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels


def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(feature,
                                                        (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load VOC dataset."""

    def __init__(self, is_train, crop_size, _voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(_voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = build_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size, _voc_dir):
    """Download and load the VOC2012 semantic dataset."""
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, _voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=0)
    return train_iter


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


model = VGG()
model = model.cuda()

summary_(model, (3, 32, 32), batch_size=1)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
cost = 0
iterations = []
train_losses = []
print("starting")
for epoch in range(150):
    model.train()
    for X, Y in load_data_voc(4, (320, 480), voc_dir):
        torch.cuda.empty_cache()
        X = X.to(cuda)
        Y = Y.to(cuda)
        optimizer.zero_grad()
        output = model(X)
        cost = loss(output, Y.long())
        cost.backward()
        optimizer.step()
        scheduler.step()

    print("Epoch : {:>4} / cost : {:>.9}".format(epoch + 1, cost))
    print("lr : {:>6}".format(scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
    iterations.append(epoch)
    train_losses.append(cost.tolist())

torch.save(model.state_dict(), './fcn vgg model/model.pt')

plt.subplot(111)
plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
plt.title('loss')
plt.show()