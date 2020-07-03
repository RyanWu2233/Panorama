## Under construction


----  
## Content
* [GAN introduction](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#gan-introduction)  
* [Generator network](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#generator-network)  
* [Discriminator network](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#discriminator-network)   
* [Loss function](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#loss-function-of-gan)  
* [Gradient penalty](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#gradient-penalty)  
* [Training tips](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#training-tips)  
* [High quality images generation](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/README.md#high-quality-images-generation)   

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


