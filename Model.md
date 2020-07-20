## Main stream GAN network ##  
 
----  
## Content
* [DCGAN](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#dcgan)  
* [Checkboard artifact](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#checkboard-artifacts)
* [Resnet](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#resnet)  
* [Self attention](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#self-attention)  
* [Self modulate](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#self-modulate)  
* [Style GAN](https://github.com/RyanWu2233/SAGAN_CelebA/blob/master/Model.md#style-gan)  


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

**Noise vector**  
* Input noise vector should be distributed around unit sphere (vector norm ~ 1). 
* Image interpolation should be computed along with unit sphere (not straight line).  
**Header**  
* Mapping to 4 x 4 resolution first.  
**For each stage**  
* Use Conv2DTranspose to upsample (kernel=(4,4), stride=2).  
* Double resolution and halves fileter number for each stage.  
* Use batch normalization to ensure convergence.  
* Use ReLU (LeakyReLU is even better).  
**Output stage**  
* Convert to RGB by Conv2D (kernel=(1,1) or (3,3), stride=2, channels=3).  
* Output image should be bounded to [-1 ~ +] by tanh function.  

***Discriminator design tips:***  

**Input image**  
* Input image should be normalized to [-1 ~ +1].  
* Real image and fake image should be trained separately.  
**For each stage**  
* Halves resolution and double fileter number for each stage.  
* Do not use batch normalization in discriminator. 
* Use Conv2D to downsample first (kernel=(4,4), stride=2, channels= f).  
* Then, use another Conv2D to smooth (kernel=(3,3), stride=1, channels= 2f).  
* Apply Leaky ReLU for after each Conv2D.  
**Output stage**
* Use Dense(1) to generate discriminator output

----  
## Checkboard artifacts  
> "Deconvolution and Checkerboard Artifacts"  
> by Augustus Odena, Vincent Dumoulin, Chris Olah, 2017 [Website](https://distill.pub/2016/deconv-checkerboard/)  

The standard approach of producing images with deconvolution — despite its successes! — has some conceptually simple issues that lead to artifacts in produced images. Conv2D and Conv2DTranspose with stride = 2 generates periodic noise pattern (n=2 for last layer, n=4 for last two layer). Using a natural alternative without these issues causes the artifacts to go away (Analogous arguments suggest that standard strided convolutional layers may also have issues):  
* The upsample Conv2DTranspose(stride=2) layer is replaced by UpSample2D(2) + Conv2D(stride=1)  
* The downsample Conv2D(stride=2) layer is replaced by Conv2D(stride=1) + AveragePooling2D(2)  

 ![Checkboard architecture](./Images/Checkboard_eff3.jpg)  

*Checkboard effect example*  
 ![Checkboard_effect_0](./Images/Checkboard_eff0.jpg)  
 ![Checkboard_effect_1](./Images/Checkboard_eff1.jpg)  
 ![Checkboard_effect_2](./Images/Checkboard_eff2.jpg)  

----
## Resnet  
> "Deep Residual Learning for Image Recognition"  
> by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2015  
> "Identity Mappings in Deep Residual Networks"  
> by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016

Resnet is proposed by He. for Imagnet classification job on 2015. It soon proves that Resnet performs well even in GAN image generation. Famous GAN architecture like SAGAN, SNGAN used Resnet architecture to improve image quality. NVIDIA Style GAN2 claims that by using Resnet for discriminator and skip-connection for generator achieves best image quality. Detail resnet architecture is shown below. Notice that resnet has several different type. Demoed one comes from NVIDIA style GAN2.  
 ![Resnet_generator](./Images/Resnet_g.jpg)  
 ![Resnet_discriminator](./Images/resnet_d.jpg)  
 
----
## Self attention  


----  
## Self modulate  


----
## Style GAN  



 
 








