## Unsupervised human face generation using TensorFlow 2.1  
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic)
![Repo celeba](https://img.shields.io/badge/Repository-CelebA-green.svg?style=plastic)
![Image size 128](https://img.shields.io/badge/Image_size-128x128-green.svg?style=plastic)  
GAN (Generative Adversial Network) is well known for high quality image generation. This project use different kind of GAN to produce 128 x 128 images using [CelebA](https://github.com/RyanWu2233/SAGAN_CelebA/tree/master/CelebA) repository. Detail analysis is made for different generator/ discriminator network architecture, and loss function and gradient penalty selection. Famous GAN architecture are shortly introduced, including: GAN, LSGAN, FGAN, WGAN, WGAN-GP, RSGAN, DRAGAN, SNGAN, SAGAN, Self-mod, PGGAN, Style GAN, and Style GAN2.  

----  
## Content
* [GAN introduction](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#gan-introduction)  
* [Loss function](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Losses.md)  

* [Generator and discriminator network](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#generator-network)  
* [High quality images generation](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#high-quality-images-generation)   
* [More result]  
* [Reference]  

----  
## GAN introduction

----  
## Generator network  
4 Generator network: GAN -> DCGAN -> Resnet -> Self_MOD -> Style_GAN -> Style_GAN2  
 

----  
## Discriminator network  
5 Discriminator network: GAN -> DCGAN -> Resnet  
 

----  
## Gradient penalty  
3 Graident penality: WGAN_GP -> R1_regularization -> PPL regularization  

----  
## Training tips  
TTUR, layer normalization, PGGAN

----  
## High quality images generation
EMA, Truncation, latent disentangle, minibatch-std, stochastic-variation
conditional batch normalization, supervised learning

----
## Reference (loss function, gradient penalty, and regularization)
***Non-saturaing GAN:*** "Generative Adversarial Nets"  
by Ian J. Goodfellow, Jean Pouget-Abadiey, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozairz, Aaron Courville, Yoshua Bengiox, 2014  

***LSGAN:*** "Least Squares Generative Adversarial Networks"  
by Xudong Mao, Qing Liy1, Haoran Xiez, Raymond Y.K. Laux, Zhen Wang, and Stephen Paul Smolley, 2015  

***WGAN:*** "Wasserstein GAN"  
by Martin Arjovsky, Soumith Chintala, and Leon Bottou, 2017  

***WGAN-GP:*** "Improved Training of Wasserstein GANs"  
by Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville, 2017  

***Hinge loss:*** "Geometric GAN"  
by Jae Hyun Lim, Jong Chul Ye, 2017  

***R1 regularization:*** "Which Training Methods for GANs do actually Converge?"  
by Lars Mescheder, Andreas Geiger, Sebastian Nowozin,  2018  










