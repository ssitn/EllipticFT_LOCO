# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:45:53 2019

@author: Sergey Sitnikov s.l.sitnikov@gmail.com
"""

import numpy as np
import os
import math
import scipy
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from operator import itemgetter
import contour as ct
#from Python_Scripts.cell_profiler_test.test.asd import contour as ct
#from asd import contour as ct


path = 'C:\\Users\\slsit\\DATA\\Python_Scripts\\cell_profiler_test\\test\\object_bin_masks'

im_arr = np.array(cv2.imread(path + '\\' + 'obj_4.tiff', cv2.IMREAD_GRAYSCALE), dtype = np.int16)
cv2.imshow('figure1', cv2.resize(np.array(im_arr, dtype=np.uint8), (800,800)))
cv2.waitKey(0)
cv2.destroyAllWindows()

im_obj = np.array(np.where(im_arr>0)).transpose()


edge = abs(np.diff(im_arr, axis = 0))
df = np.where(edge > 1) 
  
perim_idx = []
for a in range(len(df[0][:])):
    xx = (df[0][a],df[1][a])
    perim_idx.append(xx)
    
perim_line = ct.points_arrange(perim_idx) 
perim_full = ct.points_connect(perim_line, 'contour', 'unique')


An, df1, L1, A0 = ct.EllFT_coef(perim_full, 20,'loco','full')
L1 = np.array(L1)*100/np.sqrt(len(im_obj))



#asd = np.zeros((np.shape(im_arr)[0],np.shape(im_arr)[1], 4), dtype = np.int16) 
coeffs = [2, 7, 12, 17]
fig, _sub_plt = plt.subplots(2, 2, figsize=(17, 22))
sub_plt = _sub_plt.flatten()
#sub_plt_mtx = [[0,0], [0,1], [1,0], [1,1]]
for aa in range(4):
    asd = np.zeros((np.shape(im_arr)[0],np.shape(im_arr)[1]), dtype = np.int16) 
    test_perim = ct.iEllFT_coef(A0, An, coeffs[aa], perim_full, dtype = 'round_int')
    for a in range(len(perim_full)):
        asd[perim_full[a][0], perim_full[a][1]] = 255
        asd[test_perim[a][0], test_perim[a][1]] = 128
    plt.figure(1, figsize = (8.5,11))
    #sub_plt[aa].
    plt.subplot(sub_plt[aa])
    plt.subplot(sub_plt[aa]).set_title('Number of EFT \ncoefficients:'+str(coeffs[aa]), size=36)
    plt.imshow(asd[min(perim_full, key = itemgetter(0))[0]-10:max(perim_full, key = itemgetter(0))[0]+10, min(perim_full, key = itemgetter(1))[1]-10:max(perim_full, key = itemgetter(1))[1]+10], 
               cmap='viridis')
    










