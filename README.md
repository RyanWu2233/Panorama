## Panorama
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) 



----
## Image stitching
Image stiching can be direct stitching, wide angle stitching, or 360 stitching. Images should be transformed before stitching. Otherwise, significant boundary occurs at the edge of images. 
![Image_00](./JPG/PANO_00.jpg)  

----
## Image projection
Equation of image projection (transform) equation is listed below. From mathematics point of view, taking photo is equivalent to applying signal process to map 3D object onto 2D plane. 360 degree stitching (panorama) is another kind of mapping process, which maps 2D plane photo to 3D clydrical plane.
![Image_01](./JPG/PANO_01.jpg)

----
## Find descriptors
To stitch 2 photos, there must exist overlapped area between 2 adjacent photos. By applying feature extracion algorithm (SIFT, SURF, ORB, ...), descriptors of keypoints can be found.
![Image_02](./JPG/PANO_02.jpg)  

----
## Image translation, rotation, scaling and stitching
Next, descriptors between two adjacent photos are paired by KNN2 and RANSAC to find the coefficients for transformation matrix. Second photo is then translated, rotated and scaled to stitch first photo.
![Image_03](./JPG/PANO_03.jpg)  

----
## Blending
Exposure and light for each photo would not be the same. It introduces significant boundary at photo edge. This can be suppressed by using blending. However, it also blurs image at the transition band. 
![Image_04](./JPG/PANO_04.jpg)

----
## Intensity equalization
Image blend due to blending can be solved by:
(1) Apply intensity equalization before blending
(2) Use smart-cut 
![Image_05](./JPG/PANO_05.jpg)

----
## Head-to-tail connection, cropping
Another problem referred to panorama is "offset drift". It means that image chain is not connected when turn around for 360 degree. This can be solved by rotation. 
![Image_11](./JPG/PANO_11.jpg)
![Image_10](./JPG/PANO_10.jpg)

----
## Final Example
![Image_06](./JPG/PANO_06.jpg)

----
## Another example
![Image_07](./JPG/PANO_07.jpg)   




