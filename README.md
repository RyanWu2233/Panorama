## Unsupervised human face generation using TensorFlow 2.1  
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic)
![Repo celeba](https://img.shields.io/badge/Repository-CelebA-green.svg?style=plastic)
![Image size 128](https://img.shields.io/badge/Image_size-128x128-green.svg?style=plastic)  
GAN (Generative Adversial Network) is well known for high quality image generation. This project use different kind of GAN to produce 128 x 128 image sing [CelebA] repository. Detail analysis is made for different generator/ discriminator network selection, loss function and gradient penalty definition.  

----  
## Content
* [GAN introduction](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#gan-introduction)  
* [Generator network](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#generator-network)  
* [Discriminator network](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#discriminator-network)   
* [Loss function](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#loss-function-of-gan)  
* [Gradient penalty](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#gradient-penalty)  
* [Training tips](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#training-tips)  
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
## Loss function of GAN  
2 Loss function: GAN->fGAN->WGAN->DRAGAN  

----  
## Gradient penalty  
3 Graident penality: WGAN_GP -> R1_regularization -> PPL regularization  

----  
## Training tips  
TTUR, layer normalization

----  
## High quality images generation
EMA, Truncation, latent disentangle, minibatch-std, stochastic-variation
conditional batch normalization, supervised learning


