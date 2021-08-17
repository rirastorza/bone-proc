# -*- coding: utf-8 -*-
"""
2021-07-13

@author: Ramiro Irastorza

#Preprocesamiento de imagen de micro CT

"""

import matplotlib.pyplot as plt
from pylab import imread
import numpy as np
from skimage import data, color
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.transform import resize
from skimage.draw import circle
import imageio



def preprocesaMCT(imagen,center,deltaX_MCT = 15.0e-6,Lx = 2.e-3):
    """
    Centra y transforma en binaria
    """
    BONE_MCT = color.rgb2gray(imread(imagen))
    xcPix = center[0]
    ycPix = center[1]
    sampleROI_MCT = BONE_MCT[ycPix-int(Lx/deltaX_MCT):ycPix+int(Lx/deltaX_MCT),xcPix-int(Lx/deltaX_MCT):xcPix+int(Lx/deltaX_MCT)]
    thresh = threshold_otsu(sampleROI_MCT)
    binary = sampleROI_MCT > thresh
    
    loc_mean = mean(binary,disk(5))
    thresh = threshold_otsu(loc_mean)
    binary_filtrada = loc_mean > thresh
    
    #Región de interés circular
    circulo = np.zeros((len(sampleROI_MCT), len(sampleROI_MCT)), dtype=np.uint8)
    rr, cc = circle(len(sampleROI_MCT)/2,len(sampleROI_MCT)/2,int(Lx/deltaX_MCT))
    circulo[rr, cc] = 255.
   
    return binary_filtrada*circulo


pathdatos = '/home/ramiro/Documentos/CIC-2021-laptop/hueso micro modelobajafrecuencia/3D_electrical_conductivity/muestra3C/'
deltaX_MCT_muestra = 15.0e-6 # MUESTRA 2C
Lx = 2.e-3
center = [1120, 770] #sample 2c
n = 2 #reducir tamaño a la mitad
deep = 2.5e-3
NN = 580+int(deep/(deltaX_MCT_muestra))
Nimag = np.arange(580,NN,n)
mm = 100

for x in Nimag:
    imagen = pathdatos+'3c__rec0'+str(int(x))+'.bmp'
    binary_filtrada = preprocesaMCT(imagen,center, deltaX_MCT = deltaX_MCT_muestra, Lx = Lx)
    image_resized = resize(binary_filtrada, (binary_filtrada.shape[0] / n, binary_filtrada.shape[1] / n),)#anti_aliasing=True)
    imageio.imwrite('3c_binaria_norm_'+str(mm)+'.png', image_resized)
    mm = mm+1

BONE_MCT = color.rgb2gray(imread(imagen))

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
circle = Circle((center[0], center[1]),int(Lx/deltaX_MCT_muestra))
patches = []
patches.append(circle)
p = PatchCollection(patches,'red', alpha=0.4)

fig3, (ax1,ax2,ax3) = plt.subplots(ncols=3, nrows=1)#,figsize=(4, 3))
ax1.imshow(BONE_MCT, cmap=plt.cm.gray, interpolation='nearest')
ax1.add_collection(p)
ax1.set_title('MicroCT')
ax2.imshow(binary_filtrada, cmap=plt.cm.gray, interpolation='nearest')
ax2.set_title('Filtrada threshold_otsu')
ax3.imshow(image_resized, cmap=plt.cm.gray, interpolation='nearest')
ax3.set_title('Filtrada y resized')

#scipy.misc.imsave('2c_binaria_norm__rec0700.png', binary_filtrada)
plt.show()
