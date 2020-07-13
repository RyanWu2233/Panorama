## Unsupervised human face generation using TensorFlow 2.1  
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic)
![Repo celeba](https://img.shields.io/badge/Repository-CelebA-green.svg?style=plastic)
![Image size 128](https://img.shields.io/badge/Image_size-128x128-green.svg?style=plastic)  
GAN (Generative Adversial Network) is well known for high quality image generation. This project use different kind of GAN to produce 128 x 128 images using [CelebA](https://github.com/RyanWu2233/SAGAN_CelebA/tree/master/CelebA) repository. Detail analysis is made for different generator/ discriminator network architecture, and loss function and gradient penalty selection. Famous GAN architecture are shortly introduced, including: GAN, LSGAN, FGAN, WGAN, WGAN-GP, RSGAN, DRAGAN, SNGAN, SAGAN, Self-mod, PGGAN, Style GAN, and Style GAN2.  

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
Training GAN is a Min-Max game. It trains two networks simultaneously. Discriminator tends to maximum the loss while generator tends to minimum the loss. There are several kinds of loss function. Good loss function exhibit following properties: (Pg= generator distribution; Pd= reference data distribution)  
(1) Find the direction toward Pd in the begining stage.  
(2) Indicates the distance between Pg and Pd?.  
(3) Prevents from mode collapsion.  
(4) Converge to the local minimum (instead of circleing around).  
<details>
  <summary>Click to expand TensorFlow code! </summary>
  
  ## Saturation GAN (Vanilla GAN):
  
  1. A numbered
  2. list
     * With some
     * Sub bullets
     
  2 Loss function: GAN->fGAN->WGAN->DRAGAN  

</details> 

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
## Reference

[1] "Generative Adversarial Nets"  by Ian J. Goodfellow, Jean Pouget-Abadiey, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozairz, Aaron Courville, Yoshua Bengiox, 2014  




