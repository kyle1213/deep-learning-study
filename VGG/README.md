 making VGG16 with cifar-100
 
First try, lr=0.0001, epochs=10, batch size=100, last feature map is 1 * 1 *512 connected with 4096 FCL
Result was bad(test accuracy 14%). I think the last feature map size and the first FCL size gap is too big.

Second try, modified the structure of model, last feature map is 4 * 4 * 512, added BN
test accuracy 50%

Third try, changed batch size to 512, test accuracy was 50%(same with Second) but the loss was much smaller than Second
An error occurred->[ RuntimeError: CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 8.00 GiB total capacity; 6.03 GiB already allocated; 111.62 MiB free; 6.16 GiB reserved in total by PyTorch) ]
problem solved with smaller test batch size

Fourth try, changed epochs to 30
at epoch 28, the loss increased a lot
failed

Fifth try, changed epochs to 20
test accuracy 52%, highest accuracy, lowest loss, but not satisfying

Sixth try, changed epochs to 100
test accuracy 54%, lowest loss, loss changes like(5,4,3,2,1,1,1,1,1,3,2,1,1,1,1,1,2,1,1,1,1,1,1,,,,,)
