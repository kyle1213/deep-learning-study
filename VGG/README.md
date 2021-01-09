 making VGG(modified) with cifar-100

Model architecture : Conv(f=3, s=1, p=1, in=3, out=64) - BN - ReLU - Conv(f=3, s=1, p=1, in=64, out=64) - BN - ReLU - Conv(f=3, s=1, p=1, in=64, out=128) - BN - ReLU - MaxPool(k=2, s=2) - Conv(f=3, s=1, p=1, in=128, out=128) - BN - ReLU - Conv(f=3, s=1, p=1, in=128, out=128) - BN - ReLU - Conv(f=3, s=1, p=1, in=128, out=256) - BN - ReLU - MaxPool(k=2, s=2) - Conv(f=3, s=1, p=1, in=256, out=256) - BN - ReLU - Conv(f=3, s=1, p=1, in=256, out=256) - BN - ReLU - Conv(f=3, s=1, p=1, in=256, out=512) - BN - ReLU - MaxPool(k=2, s=2) - FC(4096) - FC(4096) - FC(100)

detail: loss = CEL with softmax, optimizer = Adam, lr = 0.00001, batch size = 256, epochs = 100
