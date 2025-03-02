import numpy as np
from scipy.ndimage import label

def Conectivity(img_array, min_size=25):

    '''
    Dada una imagen (en forma np array) sacamos 
    los componentes conexos de la imagen y quitamos
    aquellos menores a cierto num de pixeles.
    '''

    structure = np.ones((3, 3), dtype=np.int32) 
    labeled, num_features = label(img_array, structure)

    filtered = np.zeros_like(img_array)

    for i in range(1, num_features + 1):
        if np.sum(labeled == i) > min_size:
            filtered[labeled == i] = 255

    return filtered