## Loss function for GAN 
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic) 
  
The adversarial framework is comprised of a generator network and a discriminator network.
The generator G(z;g) maps input latent space pz(z) to fake image space Xg with parameter g. 
The discriminator output D(x) represents the probability that x came from the real imaga space (Xd)
rather than Pg. GAN train D to maximize the probability of assing correct label to both training examples and samples from G.
At same time, G is trained to minimize the cost. In other words, D and G plat the following two-player minimax game with function V(G,D).  

----  
## Content 
* [Vanilla GAN](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Losses.md#vanilla-gan)  



----  
## Vanilla GAN  
> Ref: "Generative Adversarial Nets"  by Ian J. Goodfellow, Jean Pouget-Abadiey, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozairz, Aaron Courville, Yoshua Bengiox, 2014  

![GAN_loss](./Images/Loss_eq1.png)  


----  
## LS GAN  

----  
## WGAN  


----
## WGAN GP


----
## DRAGAN  


----
## 





