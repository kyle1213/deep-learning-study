 making VGG16 with cifar-100
First, lr=0.0001, epochs=10, batch size=100, last feature map is 1 * 1 *512 connected with 4096 FCL
Result was bad(test accuracy 14%). I think the last feature map size and the first FCL size gap is too big.
Second, modified the structure of model, last feature map is 4 * 4 * 512, added BN
test accuracy 50%
Third, changed batch size to 512, test accuracy was 50%(same with Second) but the loss was much smaller than Second
An error occurred->[ RuntimeError: CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 8.00 GiB total capacity; 6.03 GiB already allocated; 111.62 MiB free; 6.16 GiB reserved in total by PyTorch) ]
solution : https://bluecolorsky.tistory.com/62, try smaller batch size
Fourth, changed epochs to 30

failed
Fifth, changed epochs to 20

can’t make test accuracy more than 50%
