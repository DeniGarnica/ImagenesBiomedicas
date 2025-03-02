import numpy as np
import matplotlib.pyplot as plt

def crea_kernel(size, tipo='disco'):

    '''
    Esta funcion crea los filtros basicos
    Ingresamos size de la regilla y tipo de filtro
    '''

    cantidad = 1

    if tipo == 'cuadrado':
        return np.ones((size, size), dtype=np.uint8) * cantidad

    if tipo == 'diamante':
        regilla = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        for i in range(size):
            for j in range(size):
                if abs(mid - i) + abs(mid - j) <= mid:
                    regilla[i, j] = cantidad
        return regilla

    if tipo == 'disco':
        regilla = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        for i in range(size):
            for j in range(size):
                if (i - mid)**2 + (j - mid)**2 <= mid**2:
                    regilla[i, j] = cantidad
        return regilla

    raise ValueError("Tipo de filtro no reconocido")


def dilatacion(img, kernel):

    '''
    Aplica dilatación a una imagen (en forma numpy array)
    en escala de grises con un kernel dado.
    '''

    k_size = kernel.shape[0]
    pad = k_size // 2
    img_padded = np.pad(img, pad, mode='edge')  # Usar padding con valores del borde
    result = np.zeros_like(img, dtype=np.uint8)

    # Aplicar dilatación
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_padded[i:i+k_size, j:j+k_size]  # Vecindad del píxel
            result[i, j] = np.max(region[kernel == 1])  # Tomar el máximo donde el kernel es 1

    return result

def erosion(img, kernel):
    '''
    Aplica erosión a una imagen (en forma numpy array)
    en escala de grises con un kernel dado.
    '''
    k_size = kernel.shape[0]
    pad = k_size // 2
    img_padded = np.pad(img, pad, mode='edge')  # Usar padding con valores del borde
    result = np.zeros_like(img, dtype=np.uint8)

    # Aplicar erosión
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_padded[i:i+k_size, j:j+k_size]  # Vecindad del píxel
            result[i, j] = np.min(region[kernel == 1])  # Tomar el mínimo donde el kernel es 1

    return result

def ver_dilataion_erosion(img, kernel):

    '''
    Dada una imagen (en forma numpy array)
    Vemos las 3 versiones
    Su original, su dilatacion y erosion
    '''

    dilated_img = dilatacion(img, kernel)
    eroded_img = erosion(img, kernel)


    # Mostrar resultados
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(dilated_img, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title("Dilatación")
    ax[1].axis("off")

    ax[2].imshow(eroded_img, cmap='gray', vmin=0, vmax=255)
    ax[2].set_title("Erosión")
    ax[2].axis("off")

    plt.show()


def TopHat(img, kernel):

    '''
    Dada una imagen (en forma numpy array)
    calculamos su TopHat = Imagen - (Dilatacion(Erosion(Imagen)))
    '''

    erode = erosion(img, kernel)
    opened = dilatacion(erode, kernel)

    result = img - opened
    return result


def BlackHat(img, kernel):
    '''
    Dada una imagen (en forma numpy array)
    calculamos su BlackHat = (Erosion(Dilatacion(Imagen))) - Imagen
    '''


    dilataded = dilatacion(img, kernel)
    closed = erosion(dilataded, kernel)

    result = closed - img
    return result

def vistas(image_original, eroded, opened, result, 
           names = ["Imagen Original", "Erosión", "Dilatación(Erosión)", "Top-Hat"]):
    
    '''
    Dadas 4 imagenes (en forma numpy array)
    Vemos estas en una sola con los nombres 
    dados en names
    '''

    # Mostrar los resultados paso a paso
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = names
    images = [image_original, eroded, opened, result]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap='gray')
        ax.set_title(title)
        ax.axis("off")