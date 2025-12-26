# -*- coding: utf-8 -*-
"""
2025-12-26

@author: Startup

#Imágenes a json de MCX

"""

import numpy as np
import zlib
import base64
import json

import numpy as np
import imageio.v2 as imageio
import os

"""
-K [1|int|str](--mediabyte)   volume data format, use either a number or a str       voxel binary data layouts are shown in {...}, where [] for byte,[i:]
       for 4-byte integer, [s:] for 2-byte short, [h:] for 2-byte half float,
       [f:] for 4-byte float; on Little-Endian systems, least-sig. bit on left
                               1 or byte: 0-128 tissue labels
                               2 or short: 0-65535 (max to 4000) tissue labels
                               4 or integer: integer tissue labels
                              96 or asgn_float: mua/mus/g/n 4xfloat format
                                {[f:mua][f:mus][f:g][f:n]}
                              97 or svmc: split-voxel MC 8-byte format
                                {[n.z][n.y][n.x][p.z][p.y][p.x][upper][lower]}
                              98 or mixlabel: label1+label2+label1_percentage
                                {[label1][label2][s:0-32767 label1 percentage]}
                              99 or labelplus: 32bit composite voxel format
                                {[h:mua/mus/g/n][s:(B15-16:0/1/2/3)(label)]}
                             100 or muamus_float: 2x 32bit floats for mua/mus
                                {[f:mua][f:mus]}; g/n from medium type 1
                             101 or mua_float: 1 float per voxel for mua
                                {[f:mua]}; mus/g/n from medium type 1
                             102 or muamus_half: 2x 16bit float for mua/mus
                                {[h:mua][h:mus]}; g/n from medium type 1
                             103 or asgn_byte: 4-byte gray-levels for mua/s/g/n
                                {[mua][mus][g][n]}; 0-255 mixing prop types 1&2
                             104 or muamus_short: 2-short gray-levels for mua/s
                                {[s:mua][s:mus]}; 0-65535 mixing prop types 1&2
       when formats 99 or 102 is used, the mua/mus values in the input volume
       binary data must be pre-scaled by voxel size (unitinmm) if it is not 1.
       pre-scaling is not needed when using these 2 formats in mcxlab/pmcx
 -a [0|1]      (--array)       1 for C array (row-major); 0 for Matlab array

"""




materials = {
    0:   {'mua': 0.001, 'mus': 1.0, 'g': 0.9,  'n': 1.0},   # aire
    100: {'mua': 0.05,  'mus': 10.0,'g': 0.9,  'n': 1.37},  # tejido 1
    200: {'mua': 0.1,   'mus': 20.0,'g': 0.95, 'n': 1.4},   # tejido 2
}

# -----------------------------
# Carpeta con los cortes
# -----------------------------
folder = "modelo1"
files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])


# -----------------------------
# Leer el primer corte para dimensiones
# -----------------------------
img0 = imageio.imread(os.path.join(folder, files[0]))
Nx, Ny = img0.shape
Nz = len(files)

print(Nz)


# -----------------------------
# Volumen de etiquetas (material)
# -----------------------------
labels = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

for k, fname in enumerate(files):
    img = imageio.imread(os.path.join(folder, fname))
    labels[:, :, k] = img

# -----------------------------
# Inicializar volumen óptico
# -----------------------------
vol = np.zeros((Nx, Ny, Nz, 4), dtype=np.float32)

# -----------------------------
# Asignar propiedades voxel a voxel
# -----------------------------
for gray_value, props in materials.items():
    mask = (labels == gray_value)
    vol[mask, 0] = props['mua']
    vol[mask, 1] = props['mus']
    vol[mask, 2] = props['g']
    vol[mask, 3] = props['n']

print("Volumen construido:")
print("Shape:", vol.shape)

#
# # Ver un corte de μa
# import matplotlib.pyplot as plt
#
# plt.imshow(vol[:, :, Nz//2, 0], cmap='inferno')
# plt.colorbar(label='μa')
# plt.title('Corte central - μa')
# plt.show()



# ========= CONFIG =========
Nx, Ny, Nz = vol.shape[:3]
spacing_unit = 0.03    # mm resolucion = 30e-6
outjson = "mcx_modelo1.json"
# ==========================

# -------- Sanity checks --------
assert vol.shape == (Nx, Ny, Nz, 4)
assert vol.dtype == np.float32

# -------- Serializar volumen --------
# MUY IMPORTANTE: order="C"
raw_bytes = vol.tobytes(order="C")

# -------- Comprimir --------
compressed = zlib.compress(raw_bytes, level=9)

# -------- Base64 --------
encoded = base64.b64encode(compressed).decode("ascii")

# -------- Armar JSON --------
mcx_json = {
    "Domain": {
        "Dim": [Nx, Ny, Nz],
        "MediaFormat": 96,
        "LengthUnit": spacing_unit
    },
    "Shapes": {
        "_ArrayType_": "float32",
        "_ArraySize_": [Nx, Ny, Nz, 4],
        "_ArrayZipType_": "zlib",
        "_ArrayZipSize_": len(compressed),
        "_ArrayZipData_": encoded
    }
}

# -------- Guardar --------
with open(outjson, "w") as f:
    json.dump(mcx_json, f, indent=2)

print("JSON MCX Cloud generado:", outjson)
print("Dim:", Nx, Ny, Nz)
print("Bytes sin comprimir:", len(raw_bytes))
print("Bytes comprimidos:", len(compressed))

