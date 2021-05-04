Studies about pure multi-layer perceptrone(MLP) or fully-connected network(FCN)

LOGs
2021 05 05 : "My own MLP" is MLP based ANN but not only manipulates width of layer but also depth of layer. This is not making MLP deeper, it's making deeper with layer's depth(1-D to 2-D). This model seems concatenating several same models(not same parameters). Have problem with broadcasting. 
2021 04 30 : Made simple MLP for MNIST in 2 ways. Using torch.nn.linear and using torch.nn.Parameter(torch.empty(in, out).cuda()), torch.nn.init.uniform_(w, a, b), torch.matmul(x, w). Both worked well
