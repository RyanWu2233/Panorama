# -*- coding: utf-8 -*-
"""
# Module name:      _Panorama_demo
# Author:           Ryan Wu
# Version:          V0.10- 2019/08/01
# Description:      Panorama demo file 
"""
import os,sys,inspect 
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------
def insert_sys_path(new_path):                      #---- Insert new path to sys
  for paths in sys.path:
    path_abs = os.path.abspath(paths);
    if new_path in (path_abs, path_abs + os.sep): return 0
  sys.path.append(new_path); 
  return 1 
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
insert_sys_path(current_dir);
parent_dir = os.path.dirname(current_dir) 
insert_sys_path(parent_dir); 
#---------------------------------------------------
from P00_CVP.CVP import CVP 
from panorama import panorama
from panorama import pano_tools
from P02a_SIFT.Features import FD_tools
pdir = parent_dir;
#=================================================== Create object and upload source images
if 1==0:                                            #
  PAN= panorama();                                  # Decalare panorama object
  PAN.upload(plt.imread('taipei_01.jpg')/256);      # Upload leftmost image
  PAN.upload(plt.imread('taipei_02.jpg')/256);      # Upload consecutive images
  PAN.upload(plt.imread('taipei_03.jpg')/256); 
  PAN.upload(plt.imread('taipei_04.jpg')/256); 
  PAN.upload(plt.imread('taipei_05.jpg')/256); 
  pano_tools.plot_source_images(PAN);               # plot source images
#=================================================== Auto execution
if 1==0:  
  pano_tools.focal_correct(PAN);
  pano_tools.extract_features(PAN);                 # extract features
  pano_tools.pairing(PAN, th= None);                # pairing feature points
  pano_tools.floorplan(PAN, tilt1=3.5);
  pano_tools.transform(PAN);
  pano_tools.registry(PAN, bw= 25, box_th= None);
  CVP.imshow(PAN.result, figy=8, ticks='off'); 
    
#=================================================== Step 1: focal length correct
if 1==0: 
  pano_tools.focal_correct(PAN);
  pano_tools.plot_focal_correct(PAN);               # plot source images   
#=================================================== Step 2: extract features 
if 1==0:
  pano_tools.extract_features(PAN);                 # extract features
  pano_tools.plot_features(PAN);                    # plot keypoints for image 0
#=================================================== Step 3: Pairing  
if 1==0:
  pano_tools.pairing(PAN, th= None);                # pairing feature points
  print('Paired feature points between image A and B:')
  print(PAN.pair_no);
#=================================================== Step 4: floor plan
if 1==0:  
  pano_tools.floorplan(PAN, tilt1=3.5);
  pano_tools.list_transform_matrix(PAN);
#=================================================== Step 5: Transform image
if 1==0:  
  pano_tools.transform(PAN);
  #pano_tools.plot_transform(PAN);  
#=================================================== Step 6: Registry
if 1==0:    
  pano_tools.registry(PAN, bw= 25, box_th= None);
  #pano_tools.deghost(PAN)
  #pano_tools.tilt_adjust(PAN)  
  #CVP.imshow(PAN.dest, figy=8, ticks='off'); 
  CVP.imshow(PAN.result, figy=8, ticks='off'); 
#=================================================== Show final result 
 
  
#===================================================    
  
  

