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

![GAN_loss_eq1](./Images/Loss_eq1.png)  

![GAN_loss_eq2](./Images/Loss_eq2.png)  
![GAN_loss_eq3](./Images/Loss_eq3.png)  
 

TensorFlow code V2.1: 
``` TensorFlow
def GAN_loss(d_real, d_fake):   
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits= True)  
    G_loss = BCE(tf.ones_like(d_fake),  d_fake)  
    D_lossR= BCE(tf.ones_like(d_real),  d_real)  
    D_lossF= BCE(tf.zeros_like(d_fake), d_fake)  
    D_loss =  D_lossR + D_lossF  
    return G_loss, D_loss  
```
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




