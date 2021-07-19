# -*- coding: utf-8 -*-
"""
# Module name:      CVP
# Author:           Ryan Wu
# Version:          V0.10- 2019/03/01
#                   V0.20- 2019/08/01
# Description:      Panorama toolkits
#
"""
import os,sys,inspect 
import pickle
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
from P02a_SIFT.SIFT import SIFT
from P02a_SIFT.Features import Keypoint
from P02a_SIFT.Features import KP_DB
from P02a_SIFT.Features import FD_tools
from P02b_ORB.ORB import ORB
""" Function call hierarchy:
    panorama- init
            - upload
    pan_tools- focal_correct
             - extract_features
             - pairing
             - floorplan
                - stitch
                  - project_p2c_points
                - derive_p2c_formula
                - remove_ill_matched_pair
                  - remove_outlier
                - linear_regression
             - transform
                - project_p2c_image
                - remove_border
                - find_edge
             - registry 
                - brightness_compensation
                  - brightness_cdf
                - paste_and_blend
                  - compute_output_size
                - cutting  
             - plot_source_images
             - plot_focal_correct
             - plot_features
             - list_transform_matrix
             - plot_registry                
"""
'=================================================== 相片接合'  
class panorama:                                     #---- Panorama
    def __init__(self):                 #---- Initialize
      #--- parameters tunable ---  
      self.hfov     = 30.8;             # Half FOV (in degrees)
      self.FD_method= 'SIFT';           # Feature detection method
      self.knn_th   = 0.5;              # pairing threshold for KNN2
      self.img1_tilt= 0;                # tilt angle for image 1
      self.band_width= 30;              # Blending band width
      self.box_th   = 0.9;              # Cutting threshold
      
      
      
      #--- parameters do not modify ----------------
      self.count    = 0;                # total image number
      self.imgs     = [];               # source images
      self.imgc     = [];               # images with focal correct
      #... extract_features
      self.KPDBs    = [];               # keypoint database
      self.DESCs    = [];               # descriptor database
      #... pairing  
      self.pair_no  = 0;                # Matrix: maching points 
                                        # between specifced images
      self.matchinfo= [];               # matching feature point info
      self.match_seq= [];               # matching image sequence
      #... floorplan and transform 
      # H=[height, width, focal, rotation, sizing, displacement Y, displacement x, err]
      self.H        = [];               # Transform function for each image
      self.imgt     = [];               # image with rotation and sizing
      self.edge     = [];               # edge of each image
      #... registry
      self.imgb     = [];               # image with brightness compensation
      self.dest     = [];               # integrate image without cutting
      self.result   = [];               # final output (with cutting)

      
      #------------------------------------------------  
      """
      self.Heq      = [];                       # 亮度校正增益曲線 [intensity_eq] 
      self.prog     = 0;                        # 自動調整的狀態
      # 0= 未開始;     1= 完成 extract;       2= 完成 pairing
      # 3= 完成 link   4= 完成 EQ,transform;  5= 完成 registry, rectify
      #-----------------------------------------# 計算用參數
      self.file_name= 'PAN3.pkl';               # 暫存檔的名稱 [save,load]
      self.IEQ      = True;                     # 亮度智能調整開關 [Intensity EQ]
      self.tilt_adjust= True;                   # 頭尾水平修正 [rectify]
      """
      
#---------------------------------------------------
    def upload(self,img):                           #---- upload source images
      self.count +=1;                               # total image count
      self.imgs.append(img);                        # add image 
      
#=================================================== Tools
class pano_tools:
#---------------------------------------------------
    def focal_correct(PAN):                         #---- project image to cylindrical
      """ imgs[k]:     kth source image without correction
          imgc[k]:     kth image with focal length correction
      """
      hfov= PAN.hfov;
      for k in range(0, PAN.count):
        src = PAN.imgs[k];
        imgc= CVP.projection_3D(src, hfov= hfov, fl=None, mode='plane_to_cylindrical'); 
        PAN.imgc.append(imgc);
#---------------------------------------------------
#---------------------------------------------------
    def extract_features(PAN, method= None):        #---- extract features
      """ KPDBs[k]:    keypoint database for image k
          DESCs[k]:    descriptor for kth image
      """
      PAN.KPDBs= [];  PAN.DESCs= [];                # clear keypoints 
      if method is not None: PAN.FD_method= method;
      if PAN.FD_method == 'ORB':    model = ORB();
      if PAN.FD_method == 'SIFT':   model = SIFT();
      model.nfeatures= 1000; 
      for k in range(0, PAN.count):
        KPDB,DESC = model.detect_and_compute(PAN.imgc[k]);  
        PAN.KPDBs.append(KPDB);                     # Add keypoint database
        PAN.DESCs.append(DESC);                     # Add descriptor database
        print('Extract image no: ',k,' keypoint number= ',KPDB.count);     
#---------------------------------------------------
#---------------------------------------------------        
    def pairing(PAN, th = None):                    #---- pairing feature points
      """ Use KNN2 to pair feature points
          pair_no[y,x]:     Total paired feature points between image x and y
          matchinfo[m]:     mth matching point information
          match_seq[m,0]:   mth matching info is related to which image
          match_seq[m,1]:   mth matching info is related to which image
      """
      if th is None: th= PAN.knn_th;                # setup KNN2 threshold
      img_no = PAN.count;
      N = int(img_no*(img_no-1)/2);                 # Total combination
      PAN.pair_no = np.zeros((img_no, img_no));     # matching point number 
      PAN.matchinfo = [];                           # Matching infomation
      PAN.match_seq = np.zeros((N, 2));
      index = 0;
      for ky in range(0, img_no-1):
          #for kx in range(ky+1, img_no):           # match to all image
          kx = ky+1;                                # only match to next image
          print('pairing between image no: ',ky,' and no: ',kx);
          match  = FD_tools.matching(PAN.DESCs[ky],PAN.DESCs[kx],th=th);
          PAN.pair_no[ky,kx] = match.shape[0];
          PAN.pair_no[kx,ky] = match.shape[0];
          PAN.matchinfo.append(match);
          PAN.match_seq[index,0] = ky; 
          PAN.match_seq[index,1] = kx;
          index += 1;
      print('Matching process complete!');
#---------------------------------------------------
#---------------------------------------------------        
    def floorplan(PAN, tilt1 = None):               #---- placement floorplan
      """ compute rotation angle, displacement, sizing relative to original
          H[k, 0]: image height (in pixel)
          H[k, 1]: image width  (in pixel)
          H[k, 2]: image focal  (in pixel)
          H[k, 3]: image rotation (in degree)
          H[k, 4]: sizing ratio
          H[k, 5]: displacement V (y-axis)
          H[k, 6]: displacement U (x-axis)
          H[k, 7]: error
      """
      if tilt1 is not None: PAN.img1_tilt = tilt1;  # setup tilt angle for image 1
      PAN.H = np.zeros((PAN.count, 8));             # Transform matrix for floorplan
      print('---- floorplanning:')
      for k in range(PAN.count):
        PAN.H[k,0]= PAN.imgc[k].shape[0];           # set image height
        PAN.H[k,1]= PAN.imgc[k].shape[1];           # set image width
        PAN.H[k,2]= (0.5/np.tan(PAN.hfov/180*np.pi))* PAN.imgs[k].shape[1];         
        PAN.H[k,4]= 1;                              # set initial sizing = 1
      PAN.H[0,3]= PAN.img1_tilt/180*np.pi;   # set tilt  
      #--- compute new floorplan for image 2 ~ ... ---
      for k in range(1, PAN.count):                 # 
      #for k in range(1, 2):                 #  debug only   
        KPS1 = PAN.KPDBs[k-1];                      # load keypoint database
        KPS2 = PAN.KPDBs[k];                        # 
        H1   = PAN.H[k-1];                          # transform function
        H2   = PAN.H[k];                            # 
        for m in range(0, PAN.match_seq.shape[0]):
          if PAN.match_seq[m, 0]== (k-1) and PAN.match_seq[m, 1]== k:             
            match = PAN.matchinfo[m];
            pano_tools.stitch(KPS1, KPS2, H1, H2, match); 
#---------------------------------------------------
    def stitch(KPS1, KPS2, H1, H2, match):          #---- stich image to previous one
      """ Project image1 [R1,C1] to cylindrical [V1,U1]
          and then derived H2 for image 2 by:
          [V1,U1]= [V2,U2]= H2[R2,C2]    
      """
      #--- projection image1 from plane to cylindrical ---
      total  = np.minimum(match.shape[0],100);      # total pairing number
      bin1   = match[0:total,0].astype(int);        # feature no at image 1
      R1     = KPS1.keyz[bin1, 0];                  # keypoint Y at image 1
      C1     = KPS1.keyz[bin1, 1];                  # keypoint X at image 1
      V1, U1 = pano_tools.project_p2c_points(R1, C1, H1);
      #--- image 2 ---
      bin2   = match[0:total,1].astype(int);        # feature no at image 2
      R2     = KPS2.keyz[bin2, 0];                  # keypoint Y at image 2
      C2     = KPS2.keyz[bin2, 1];                  # keypoint X at image 2
      Rc2    = H2[0]/2;  Rp2= R2 - Rc2; 
      Cc2    = H2[1]/2;  Cp2= C2 - Cc2;
      #--- --- 
      # {phi1,S1,TU1,TV1} = M*M matrix: which is derived by chosen 2 pairs 
      # {phi0,S0,TU0,TV0} = scalar: which is initial guess by removing outlier
      # 
      phi1,S1,TU1,TV1= pano_tools.derive_p2c_formula(U1,V1,Cp2,Rp2);
      seq,phi0,S0,TU0,TV0 = pano_tools.remove_ill_matched_pair(phi1,S1,TU1,TV1);      
      #--- linear regression [not necessary] ---
      # U1X = U1[seq];  C2X = C2[seq]; V1X = V1[seq];  R2X = R2[seq];  
      # phi0,S0,TU0,TV0,Err= pano_tools.linear_regression(V1X,U1X,R2X,C2X, phi0,S0,TU0,TV0,H2)
      H2[3]= phi0; H2[4]= S0; H2[5]= TV0; H2[6]= TU0;      
#---------------------------------------------------      
    def linear_regression(V1,U1,R2,C2,Phi,S,TU,TV,H2):
      DW= 1e-9; LR= 0.01;
      total = V1.shape[0];
      for k in range (0,3000):
        HX     = np.array([H2[0],H2[1],H2[2],Phi,S,TV,TU,0]);
        HU     = np.array([H2[0],H2[1],H2[2],Phi,S,TV,TU+DW,0]);
        HV     = np.array([H2[0],H2[1],H2[2],Phi,S,TV+DW,TU,0]);
        HS     = np.array([H2[0],H2[1],H2[2],Phi,S+DW,TV,TU,0]);
        HP     = np.array([H2[0],H2[1],H2[2],Phi+DW,S,TV,TU,0]);
        #--- ---
        VX,UX  = pano_tools.project_p2c(R2,C2,HX);     
        VU,UU  = pano_tools.project_p2c(R2,C2,HU);    # 使用新轉換參數推導結果
        VV,UV  = pano_tools.project_p2c(R2,C2,HV);    # 使用新轉換參數推導結果
        VS,US  = pano_tools.project_p2c(R2,C2,HS);    # 使用新轉換參數推導結果
        VP,UP  = pano_tools.project_p2c(R2,C2,HP);    # 使用新轉換參數推導結果 
        #--- 計算 cost function ---
        JX     = np.sqrt(np.sum((VX-V1)**2+(UX-U1)**2))/total;
        JU     = np.sqrt(np.sum((VU-V1)**2+(UU-U1)**2))/total;
        JV     = np.sqrt(np.sum((VV-V1)**2+(UV-U1)**2))/total;
        JS     = np.sqrt(np.sum((VS-V1)**2+(US-U1)**2))/total;
        JP     = np.sqrt(np.sum((VP-V1)**2+(UP-U1)**2))/total;

        #if np.mod(k,1000)==0: print(k,JX, Phi,S,TV,TU)  # Debug only
        #--- 計算 梯度 ---
        GRAD_U = (JU-JX)/DW;  TU  = TU  -LR*GRAD_U;
        GRAD_V = (JV-JX)/DW;  TV  = TV  -LR*GRAD_V;
        GRAD_S = (JS-JX)/DW;  S   = S   -LR*GRAD_S/1000;
        GRAD_P = (JP-JX)/DW;  Phi = Phi -LR*GRAD_P/1000;
      #--- --- 
      Err = np.round(JX*100)/100;
      return Phi, S, TU, TV, Err
 #---------------------------------------------------      
    def remove_ill_matched_pair(phi1,S1,TU1,TV1):   #---- remove ill matched pair
      """ check all combinations and remove unreliable pair
      """
      #--- mark inlier= 1; outlier= 0 ---
      mask, phi0= pano_tools.remove_outlier(phi1);
      mask, S0  = pano_tools.remove_outlier(S1  ,Nstd=2, mask= mask);
      mask, TU0 = pano_tools.remove_outlier(TU1 ,Nstd=2, mask= mask);
      mask, TV0 = pano_tools.remove_outlier(TV1 ,Nstd=2, mask= mask);       
      mask, phi0= pano_tools.remove_outlier(phi1,Nstd=3, mask= mask);
      mask, S0  = pano_tools.remove_outlier(S1  ,Nstd=3, mask= mask);
      mask, TU0 = pano_tools.remove_outlier(TU1 ,Nstd=3, mask= mask);
      #--- select reliable data pair ---
      # mask is M*M matrix: 1= reliable pair combination;
      M   = phi1.shape[0];
      sumx= np.sum(mask,axis=0);                    # large number= reliable
      seq = [];                                     # chosen reliable data
      for k in range(0, int(M*0.7)):
        maxx = np.argmax(sumx);
        seq.append(maxx);
        sumx[maxx]= 0; 
      return seq, phi0, S0, TU0, TV0
#---------------------------------------------------      
    def remove_outlier(data, Nstd=2, mask= None):   #---- remove extreme data
      """ data is N*N matrix
          mask is N*N matrix
          if mask== None: create blank mask (1=inside, 0= outside)
      """
      M = data.shape[0]; 
      if mask is None:
        mask = np.ones((M,M));                      # if mask not existed
        for k in range(0,M): mask[k,k]= 0;          # create one and remove diagnol
      N = np.sum(mask);                             # total effective data number   
      sumx= np.sum(data* mask);
      mean= sumx/ N;                                # new mean
      sum_square = np.sum(((data-mean)*mask)**2);   #
      std = np.sqrt( sum_square/ (N-1) );           # new standard deviation
      #--- ---
      larger = data > (mean+ Nstd*std);             # data too large
      smaller= data < (mean- Nstd*std);             # data too small
      maskh  = mask.copy();
      maskh[larger] = 0;  maskh[smaller]= 0;        # remove outlier data
      return maskh, mean       
#---------------------------------------------------      
    def derive_p2c_formula(U1,V1,Cp2,Rp2):          #---- derive p2c formula
      """ Derive p2c formula:
            U= s*[cosf*(C-Cc)- sinf*(R-Rc)] + tu
            V= s*[sinf*(C-Cc)+ cosf*(R-Rc)] + tv
            
          by following equations:
          phi= arctan2(DV*DC-DU*DR, DU*DC+DV*DR);
            s= DU/(cosf*DC- sinf*DR);
           tu= Ua- s*[cosf*(C-Cc)- sinf*(R-Rc)]
           tv= Va- s*[sinf*(C-Cc)+ cosf*(R-Rc)]
           
         Use 100 pairs and check all 4950 combinations
      """
      total  = U1.shape[0];
      phi    = np.zeros((total, total));            # rotation angle (0~pi)
      S      = np.zeros((total, total));            # sizing ratio
      TU     = np.zeros((total, total));            # displacement along X
      TV     = np.zeros((total, total));            # displacement along Y      
      for ky in range(0, total):
        DU = U1[ky] - U1;   DC = Cp2[ky] - Cp2;
        DV = V1[ky] - V1;   DR = Rp2[ky] - Rp2;
        phi[ky,:] = np.arctan2(DV*DC-DU*DR, DU*DC+DV*DR);
        cosf      = np.cos(phi[ky,:]);
        sinf      = np.sin(phi[ky,:]);
        #--- prevent from divide by 0 ---
        denom     = (cosf*DC- sinf*DR);
        denom[denom==0]= DU[denom==0];
        denom[denom==0]= 1;         
        S[ky,:]   = DU/denom;
        #--- ---
        TU[ky,:]  = U1- S[ky,:]*(cosf*Cp2- sinf*Rp2);
        TV[ky,:]  = V1- S[ky,:]*(sinf*Cp2+ cosf*Rp2);   
      return phi,S,TU,TV  
#---------------------------------------------------
    def project_p2c_points(R, C, H):                #---- project to cylindrical
      """ project plane to cylindrical
          R: Y of plane coordinates
          C: X of plane coordinates
          H: Transform matrix for floorplaning
          return:
          V: Y of cylindrical coordinates
          U: X of cylindrical coordinates          
      """
      Rc= H[0]/2; Cc= H[1]/2;                       # center coordinate
      phi = H[3];  S= H[4];                         # rotation angle and sizing
      Tv  = H[5]; Tu= H[6];                         # displacement
      COSF= np.cos(phi);  SINF= np.sin(phi);          #  
      U  = Tu + S*( COSF*(C- Cc)- SINF*(R- Rc) ); 
      V  = Tv + S*( SINF*(C- Cc)+ COSF*(R- Rc) );
      return V, U   
#---------------------------------------------------
#---------------------------------------------------
    def transform(PAN):                             #---- transform images to cylindrical
      #--- FOV-> sizing+ rotation + minor displacement ---
      PAN.imgt= [];  
      for k in range(0, PAN.count):
        img = pano_tools.project_p2c_image(PAN.imgs[k],PAN.H[k]);
        img = pano_tools.remove_border(img);
        PAN.imgt.append(img); 
      pano_tools.find_edge(PAN);
      
#---------------------------------------------------
    def project_p2c_image(src, H):                  #---- project p to c (whole image)
      """ Integrate following process to reduce quality loss
          during multiple transformation
         [Forward transform: V,U <-- R,C]
           theta= arctan(C/Z);
           C'= Z*theta;  R'= R*cos(theta);
           U = dTU+ S*[cosf*C' - sinf*R']
           V = dTV+ S*[sinf*C' + cosf*R'] 
           
         [Backward transform: R,C <-- V,U]
           U'= (U-dTU)/S;  V'= (V-dTV)/S;
           C'= [ cosf*U' + sinf*V']
           R'= [-sinf*U' + cosf*V']
           theta= (C'/Z);
           C= Z*tan(theta);   
           R= R'/cos(theta);
           C= C'/cos(theta)* [sin(theta)/theta];            
      """
      Z   = H[2]; phi= H[3]; S= H[4]; TV= H[5]; TU= H[6];
      rows= src.shape[0]; cols= src.shape[1];       # get image size info
      diag= np.sqrt(rows**2+cols**2);               # diagnol length
      radi= int(diag*S/2*1.1);                      # radius of new plot should be larger
      dest= np.zeros((radi*2,radi*2,3))             # projection result
      cosf= np.cos(phi); sinf= np.sin(phi);         # rotation parameters
      u0  = radi-(TU-np.floor(TU));                 # only process fractional part
      v0  = radi-(TV-np.floor(TV));                 # of TU and TV
      kv  = np.arange(0,radi*2);                    # 
      #--- ---
      srcx= src.copy();
      srcx[0,:,:]=0; srcx[rows-2:rows,:,:]=0; 
      srcx[:,0,:]=0; srcx[:,cols-2:cols,:]=0;
      #--- mapping ---
      for ku in range(0,radi*2):                    # scan each column
        UP = (ku-u0)/S;  VP= (kv-v0)/S;             # correct tu,tv,s
        RP =-sinf*UP + cosf*VP;
        CP = cosf*UP + sinf*VP;                     # correct rotation phi
        theta= CP/Z;                                # horizontal angle
        C  = Z*np.tan(theta)  + cols/2;
        R  = RP/np.cos(theta) + rows/2;
        #--- interpolation ---
        C  = np.minimum(np.maximum(C, 0), cols-2);
        R  = np.minimum(np.maximum(R, 0), rows-2); 
        C0 = np.floor(C).astype(int); C1= C-C0; 
        R0 = np.floor(R).astype(int); R1= R-R0; 
        for m in range(0,3):
          pixel = srcx[R0  ,C0  ,m]*(1-R1)*(1-C1);
          pixel+= srcx[R0  ,C0+1,m]*(1-R1)*(  C1);
          pixel+= srcx[R0+1,C0  ,m]*(  R1)*(1-C1);
          pixel+= srcx[R0+1,C0+1,m]*(  R1)*(  C1);
          dest[kv,ku,m]= pixel;          
      return dest
#---------------------------------------------------
    def remove_border(src):                         #---- remove blank border
      """ remove black area on 4 edges
      """
      rows = src.shape[0]; VMIN= 0; VMAX= rows; 
      cols = src.shape[0]; UMIN= 0; UMAX= cols;
      for ky in range(1,rows):
        sum0 = np.sum(src[ky,:,:]);
        sum1 = np.sum(src[rows-ky-1,:,:]);
        if sum0== 0 and VMIN== ky-1:      VMIN= ky;
        if sum1== 0 and VMAX== rows-ky+1: VMAX= rows-ky;
      for kx in range(1,cols):
        sum0 = np.sum(src[:,kx,:]);
        sum1 = np.sum(src[:,cols-kx-1,:]);
        if sum0== 0 and UMIN== kx-1:      UMIN= kx;
        if sum1== 0 and UMAX== cols-kx+1: UMAX= cols-kx;
      #--- ---  
      DV = np.minimum(VMIN, rows-VMAX);
      DU = np.minimum(UMIN, cols-UMAX);
      return src[DV:(rows-DV), DU:(cols-DU), :]; 
#---------------------------------------------------
    def find_edge(PAN):                             #---- Find image edge
      PAN.edge = [];
      for k in range(0, PAN.count):
        src = PAN.imgt[k];
        rows= src.shape[0]; cols= src.shape[1];
        edge= np.zeros((rows,2));
        for ky in range(0,rows):                    #.. scan each row
          kx= 0;  
          while np.sum(src[ky,kx,:])==0 and kx<cols-1: kx+=1;     #. find left border
          edge[ky,0]= kx+1;  
          kx= cols-1;  
          while np.sum(src[ky,kx,:])==0 and kx>edge[ky,0]: kx-=1; #. find right border
          edge[ky,1]= kx-1;  
        PAN.edge.append(edge);
#--------------------------------------------------- 
            
#---------------------------------------------------        
#---------------------------------------------------
    def registry(PAN, bw= None, box_th= None):      #---- paste image to final output
      """ [1] Copy and paste
          [2] Blending
          [3] Cutting 
          bw:   Blending width
      """
      # Get reference brightness
      PAN.imgb=[];
      for k in range(0,PAN.count): PAN.imgb.append(PAN.imgt[k]);
      print('Brightness compensation')
      pano_tools.brightness_compensation(PAN)
      pano_tools.brightness_compensation(PAN)
      print('Paste and blend')
      PAN.dest = pano_tools.paste_and_blend(PAN, src=PAN.imgb, bw= bw);        
      pano_tools.cutting(PAN, box_th=box_th);       # cutting            
#---------------------------------------------------
    def compute_output_size(PAN):                   #---- compute output size
      imgt= PAN.imgt;                               # 
      H   = PAN.H.copy();                           # Transform matrix
      H[:,5]= np.floor(H[:,5]); 
      H[:,6]= np.floor(H[:,6]);
      #--- ---
      xa  = np.argmin(H[:,6]); x0  = np.ceil(imgt[xa].shape[1]/2);
      xb  = np.argmax(H[:,6]); x1  = np.ceil(imgt[xb].shape[1]/2); 
      x2  = H[xb,6] - H[xa,6];
      cold= int(x0+ x1+ x2)
      ya  = np.argmin(H[:,5]); y0  = np.ceil(imgt[ya].shape[0]/2);
      yb  = np.argmax(H[:,5]); y1  = np.ceil(imgt[yb].shape[0]/2); 
      y2  = H[yb,5] - H[ya,5];
      rowd= int(y0+ y1+ y2)
      return x0, y0, cold, rowd
#---------------------------------------------------
    def paste_and_blend(PAN, src, bw= None):        #---- paste and blending
      # src = PAN.imgt 
      x0, y0, cold, rowd = pano_tools.compute_output_size(PAN);
      dest= np.zeros((rowd,cold,3));                # final output
      if bw is None: bw= PAN.band_width;
      bwr = np.arange(0,bw)/bw;
      H   = PAN.H.copy();                           # Transform matrix
      #--- ---      
      for k in range(0, PAN.count):                 #... scan each image
        #src = PAN.imgt[k];
        rows= src[k].shape[0]; cols= src[k].shape[1];
        y_off  = int(H[k,5] + y0 - rows/2);         # offset between original image
        x_off  = int(H[k,6] + x0 - cols/2);         # and final output
        for ky in range(0, rows):                   #.. scan each row
         if PAN.edge[k][ky,0]<PAN.edge[k][ky,1]:
          dy  = int(ky + y_off);                    # destination row
          #--- paste direct copy area ---
          sx0 = int(PAN.edge[k][ky,0])+bw;  dx0 = sx0 + x_off;
          sx1 = int(PAN.edge[k][ky,1]);     dx1 = sx1 + x_off;
          dest[dy,dx0:dx1,:]= src[k][ky,sx0:sx1,:];
          #--- paste blending area ---
          for kx in range(0,bw):
            sx = int(PAN.edge[k][ky,0])+kx; dx  = sx  + x_off;
            if np.sum(dest[dy,dx,:])==0: ratio=1;
            else: ratio= bwr[kx];             
            dest[dy,dx,:]= dest[dy,dx,:]*(1-ratio)+ src[k][ky,sx,:]*ratio;        
      #--- ---
      return dest;           
#---------------------------------------------------
    def cutting(PAN, box_th= None):                 #---- cut outlier region
      """ find top and bottom boundary by 95% non-blank
          then, find left and right boundary
      """
      if box_th is None: box_th = PAN.box_th;       # cutting threshold
      src = PAN.dest; 
      rows= src.shape[0]; cols= src.shape[1];       # get un-cut image size
      #--- find top boundary ---
      top = 0; flag= 0;
      while flag==0:
        inactive= np.sum(np.sum(src[top,:,:], axis=1)>0);
        if inactive> box_th*cols or top==rows: flag=1;
        else: top+=1;
      #--- find bottom boundary ---
      bottom= rows-1; flag=0;
      while flag==0:
        inactive= np.sum(np.sum(src[bottom,:,:], axis=1)>0);
        if inactive> box_th*cols or bottom==0: flag=1;
        else: bottom -=1;
      #--- find left boundary ---
      rowa= bottom- top;
      left = 0; flag=0;
      while flag==0:
        inactive= np.sum(np.sum(src[top:bottom,left,:], axis=1)>0);
        if inactive> box_th*rowa or left==cols: flag=1;
        else: left+=1;
      #--- find right boundary ---
      right= cols-1; flag=0;
      while flag==0:
        inactive= np.sum(np.sum(src[top:bottom,right,:], axis=1)>0);
        if inactive> box_th*rowa or right== 0: flag=1;
        else: right -=1;
      PAN.result= PAN.dest[top:bottom, left:right, :];
#---------------------------------------------------
#---------------------------------------------------     
    def brightness_compensation(PAN):               #---- brightness compensation
      #--- reference (no boundary but blurred) ---
      ref = pano_tools.paste_and_blend(PAN, src= PAN.imgb, bw=70);
      gref= CVP.rgb2gray(ref);
      H   = PAN.H.copy();  
      xa  = np.argmin(H[:,6]); x0  = int(np.ceil(PAN.imgb[xa].shape[1]/2));
      ya  = np.argmin(H[:,5]); y0  = int(np.ceil(PAN.imgb[ya].shape[0]/2));
      box_ratio = 0.99;
      #--- ---
      for k in range(0,PAN.count):          
        #--- capture source sampling window ---  
        imgb= CVP.rgb2gray(PAN.imgb[k]); 
        rowh= int(imgb.shape[0]/2);                 # half image height
        colh= int(imgb.shape[1]/2);                 # half image width        
        roww= int(rowh* box_ratio);                 # capture window radius
        colw= int(colh* box_ratio);
        imgs= imgb[rowh-roww:rowh+roww, colh-colw:colh+colw]
        cdfs= pano_tools.brightness_cdf(imgs);      # brightness curve for source         
        #--- capture reference sampling window ---  
        yr  = int(y0+ H[k,5]);
        xr  = int(x0+ H[k,6]);
        imgr= gref[yr-roww:yr+roww, xr-colw:xr+colw];
        cdfr= pano_tools.brightness_cdf(imgr);      # brightness curve for reference         
        #---  brightness equalization ---
        out = np.zeros_like(PAN.imgb[k]);           # image with brightness compensation
        for kb in range(0,256):
          under= np.sum(cdfr< cdfs[kb]);            # 
          under= np.minimum(under, 3*(kb+0.5)-0.5); # prevent from low gain boost
          gain = (under+0.5)/(kb+0.5); 
          #--- find pixel---
          th0  = kb/256; th1 = (kb+1)/256;          # bin lower and upper bound
          mask = ((imgb<th1)+0) - ((imgb<=th0)+0)   # data inside the region
          mask = mask* gain;
          out[:,:,0]+= PAN.imgt[k][:,:,0]*mask;
          out[:,:,1]+= PAN.imgt[k][:,:,1]*mask; 
          out[:,:,2]+= PAN.imgt[k][:,:,2]*mask; 
        PAN.imgb[k] = out;          
 #---------------------------------------------------      
    def brightness_cdf(src):                        #---- compute brightness curve
      pdf = np.zeros(256);                          # probability density function
      rows= src.shape[0]; cols= src.shape[1];
      for ky in range(0,rows):
        for kx in range(0,cols):  
          bins = int(np.round(src[ky,kx]*255)); pdf[bins] +=1;
      cdf = np.zeros(256); cdf[0]= pdf[0];          # cumulation density function
      for kb in range(1,256):
        cdf[kb] = cdf[kb-1]+ pdf[kb];
      cdf= cdf/(rows*cols);
      return cdf
#---------------------------------------------------      
    
  
      
#===================================================

#---------------------------------------------------
    def plot_source_images(PAN) :                   #---- plot source image base
      for k in range(0, np.shape(PAN.imgs)[0]): 
        CVP.imshow(PAN.imgs[k],figy=6,ticks='off');
        print('image no:',k);
#---------------------------------------------------
    def plot_focal_correct(PAN):                    #---- plot source image base      
      for k in range(0, np.shape(PAN.imgc)[0]): 
        CVP.imshow(PAN.imgc[k],figy=6,ticks='off');
        print('image no:',k);
#---------------------------------------------------    
    def plot_features(PAN):                         #---- plot features for each image      
      for k in range(0, np.shape(PAN.imgc)[0]): 
        FD_tools.plot_keypoints(PAN.imgc[k], PAN.kps[k],
                                keytype='arrow', keycolor='b',
                                figwidth=8, keysize =2); 
        print('image no:',k);
#---------------------------------------------------
    def list_transform_matrix(PAN):                 #---- list transform matrix
      print('-----------------------------------------------------')
      print(' height,width,focal,angle,sizing,   T(Y),   T(X)')
      for k in range(0, PAN.H.shape[0]):
        H= PAN.H[k,:];  
        rows = int(H[0]);  cols= int(H[1]);  focal= int(H[2]);
        angle= int((H[3]*180/np.pi)*100)/100;
        size = int( H[4]*100)/100;
        TV   = int( H[5]*100)/100;
        TU   = int( H[6]*100)/100;
        txt  = '%6d %5d %6d %5.2f %5.2f %8.2f %8.2f' % (rows,cols,focal,angle,size,TV,TU)
        print(txt);  
#---------------------------------------------------
    def plot_transform(PAN):                        #---- plot registry images
      for k in range(0, np.shape(PAN.imgt)[0]):
        CVP.imshow(PAN.imgt[k],figy=6,ticks='off');
        print('image no:',k);
        
#---------------------------------------------------
#---------------------------------------------------
