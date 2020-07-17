## Main stream network for GAN ##  

----  
## Content
* [DCGAN]()  
* [Resnet]()  
* [Self attention]()  
* [Self modulate]()  
* [Style GAN]()  


----  
## DCGAN
> "Unsupervised representation learning with deep convolutional GAN"  
> by Alec Radford & Luke Metz, Soumith Chintala, 2016  

When Ian.Goodfellow first proposed GAN netowrk on 2014, everybody is exciting for it's elegant concept and capability. 
But they soon found that GAN is quite difficult to train. 
On 2016, DCGAN provides CNN based structure which is robust and easy to train. 
Until now, almost all GAN image generation works follow DCGAN's structure.  
DCGAN first maps the 128 dimensional noise tensor into 4x4 low resolution image with 1024 filters. 
Then the resolution is up-sampled by 2 and filter number halves in every stage. 
In final stage, convolution layer with channels=3 is used to map the final output to RGB space. 
Tanh activation layer is used to clip output in the range [-1 ~ +1].
It is then mapping to 0~256 for image display purpose.
 ![network_DCGAN1](./Images/mdl_dcgan1.jpg)  

Detail network for generator and discriminator are shown below:
 ![network_DCGAN](./Images/mdl_dcgan.jpg)  
 
***Generator design tips:***  
*Noise vector*  
* Input noise vector should be distributed around unit sphere (vector norm ~ 1). 
* Image interpolation should be computed along with unit sphere (not straight line).
*Header*  
* Mapping to 4 x 4 resolution first.  
*For each stage*  
* Use Conv2DTranspose to upsample (kernel=(4,4), stride=2).  
* Use batch normalization to ensure convergence.  
* Use ReLU (LeakyReLU is even better).  
* Double resolution and halves fileter number for each stage.  
*Output stage*
* Convert to RGB by Conv2D (kernel=(1,1) or (3,3), stride=2, channels=3).  
* Output image should be bounded to [-1 ~ +] by tanh function.  

***Discriminator design tips:***

* Input image should be normalized to [-1 ~ +1].
* Real image and fake image should be classified separately.
* Do not use batch normalization in discriminator. 
* Use Conv2D to downsample (stride=2)











