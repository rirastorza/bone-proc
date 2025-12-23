# -*- coding: utf-8 -*-
"""
2021-07-13

@author: Ramiro Irastorza

#Preprocesamiento de imagen de micro CT

"""
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.draw import disk as draw_disk
from skimage.transform import resize
import imageio

def preprocesaMCT(imagen, center, deltaX_MCT=15.0e-6, Lx=2.e-3):
    """
    Centra la imagen MicroCT, binariza y aplica máscara circular.
    Compatible con skimage >= 0.22
    """

    # --- Leer imagen y convertir a grayscale ---
    img = imread(imagen)

    if img.ndim == 3:
        img = img[..., :3]          # eliminar alpha si existe
        BONE_MCT = rgb2gray(img)
    else:
        BONE_MCT = img.astype(float)

    xcPix, ycPix = int(center[0]), int(center[1])
    R = int(Lx / deltaX_MCT)

    # --- ROI cuadrada centrada ---
    sampleROI_MCT = BONE_MCT[
        ycPix - R : ycPix + R,
        xcPix - R : xcPix + R
    ]

    # --- Umbral global ---
    thresh = threshold_otsu(sampleROI_MCT)
    binary = sampleROI_MCT > thresh

    # --- rank.mean NO acepta bool → uint8 ---
    binary_u8 = (binary.astype(np.uint8) * 255)

    loc_mean = mean(binary_u8, disk(5))

    thresh2 = threshold_otsu(loc_mean)
    binary_filtrada = loc_mean > thresh2

    # --- Máscara circular ---
    h, w = binary_filtrada.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    rr, cc = draw_disk((h // 2, w // 2), R)
    mask[rr, cc] = 1

    return binary_filtrada * mask
#
#
pathdatos = '/home/ramiro/Documentos/Startup/Imagenes/photoacoustic/modelos realistas/Micro CT - YTEC - abril 2018/3c/3c_Rec/'
deltaX_MCT_muestra = 15.0e-6 # MUESTRA 2C
Lx = 6.e-3
center = [1250, 1000] #sample 2c
n = 2 #reducir tamaño a la mitad
deep = 4.e-3
NN = 580+int(deep/(deltaX_MCT_muestra))
Nimag = np.arange(580,NN,n)
mm = 100

for x in Nimag:
    imagen = pathdatos+'3c__rec0'+str(int(x))+'.bmp'
    binary_filtrada = preprocesaMCT(imagen,center, deltaX_MCT = deltaX_MCT_muestra, Lx = Lx)
    # image_resized = resize(binary_filtrada, (binary_filtrada.shape[0] / n, binary_filtrada.shape[1] / n),)#anti_aliasing=True)

    image_resized = resize(binary_filtrada,
    (binary_filtrada.shape[0] // n,
     binary_filtrada.shape[1] // n),
    anti_aliasing=False)
    image_out = (image_resized > 0).astype(np.uint8) * 255

    imageio.imwrite('3c_binaria_norm_'+str(mm)+'.png', image_out)
    mm = mm+1

from skimage import color

BONE_MCT = color.rgb2gray(imread(imagen))

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

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
