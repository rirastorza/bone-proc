# bone-proc
Procesamiento digital de imágenes de huesos

En este tutorial se comenta una manera de construir modelos para resolver con elementos finitos a partir de mediciones tomográficas de hueso. En particular, mediciones con microCT de hueso trabecular bovino. 

Está dividido en tres partes:

1. [Preprocesamiento de imágenes, decimación y corrección de malla, y finalmente construcción del archivo .stl](https://github.com/rirastorza/bone-proc/blob/main/filtrado/funciones_filtrado.ipynb). Hay unos videos explicativos donde se utiliza 3DSlicer y Meshlab.
2. [Construcción de malla con Gmsh](https://github.com/rirastorza/bone-proc/blob/main/malla_mef/generacion_malla.ipynb).
3. Resolución con MEF utilizando FEniCS.

Los scripts completos están en la carpeta [scripts](https://github.com/rirastorza/bone-proc/tree/main/scripts).

### ¿Cómo usar?

- Clonar o bajar el repositorio.
- Utilizar las jupyter notebook para seguir los pasos.

---

## References

1. Algunos paquetes de Python: NumPy, Scipy, [Scikit-image](https://scikit-image.org/), [Matplotlib](https://matplotlib.org/).
2. El modelo 3D de la superficie es construido utilizando [3DSlicer](https://www.slicer.org/).
3. La malla inicial es preprocesada utilizando [Meshlab](https://www.meshlab.net/).
4. La malla es generada con [Gmsh](https://gmsh.info/), copyright (C) 1997-2020 by C. Geuzaine and J.-F. Bajo GNU General Public License (GPL).
5. [FEniCS](https://fenicsproject.org/download/) open-source (LGPLv3) para resolver ecuaciones en derivadas parciales utilizando el Método de Elementos Finitos.
6. Para análisis y visualización se utiliza [ParaView](https://www.paraview.org/), licencia BSD.
