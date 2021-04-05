# deep-learning-study
these are my deeplearning presetations for seminar and summary notes

papers I read : ImageNet Classification with Deep Convolutional Neural Networks - AlexNet / used two GPU for learning and matching model(divided by two)
                Very Deep Convolutional Networks for Large-Scale Image Recognition - VGG net / one 5x5 conv and two 3x3 conv have same receptive fields, but 3x3 is deeper and has less parameters to learn
                Going deeper with convolutions - GoogLeNet / used inecption module / inception module uses 1x1 conv to change the number of channels and reduced amount of calculations
                Deep Residual Learning for Image Recognition - ResNet / used skip connection(short cut) / deep networks had gradient vanishing problem, but with residual connection, can go deeper 
                Fully Convolutional Networks for Semantic Segmentation - FCN / no fully connected layer because flattening feature maps may lose their local information and it's not good for semantic segmentation tasks / used 1x1 conv instead of fully connected layer
                U-Net: Convolutional Networks for Biomedical Image Segmentation - UNet
                Learning Deconvolution Network for Semantic Segmentation! - Deconvolutional Networks / used deconv many times, upsampling feature maps little by little
                Attention is All You need : Transformer / used multi head self attention / no rnn based methods used, only attention and fully connected layer
               
            
