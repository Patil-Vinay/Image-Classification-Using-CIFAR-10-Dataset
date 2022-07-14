# Image-Classification-Using-CIFAR-10-Dataset
CIFAR-10 is an image dataset which contains 60000 tiny color images with the size of 32 by 32 pixels. The dataset consists of 10 different classes (i.e. airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck), in which each of those classes consists of 6000 images. I have used CNN aka Convolution Neural Network to train the model. Used multiple layers in the model to get better accuracy. Parameter tuning playes a vital role in training the model.

First I trained the model without using data augmentation the graph for accuracy and validation accuracy is as follows: 

![alt text](https://github.com/Patil-Vinay/Image-Classification-Using-CIFAR-10-Dataset/blob/main/Result%20without%20using%20Data%20Augmentation.png)

Then I applied data augmentation and the curve which I got is as folows:

![alt text](https://github.com/Patil-Vinay/Image-Classification-Using-CIFAR-10-Dataset/blob/main/Result%20using%20Data%20Augmentation.png)

As we can see the gap between validation accuracy and accuracy is more in model which is trained without data augmentation. 

Basically the gap between training and validation accuracy is a clear indication of overfitting. The larger the gap, the higher the overfitting.
