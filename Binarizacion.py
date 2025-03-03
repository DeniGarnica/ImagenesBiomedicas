
#------------------#
#-------OTSU-------#
#------------------#

# Implementado por Denisse Garnica c:

'''
EL algoritmo busca exhaustivamente el umbral
que minimiza la intra-class variance
Separando entre dos calses (fondo/objeto)
'''


import numpy as np

def otsu_threshold(img, verbose = 1):
    """
    Encontrar el umbral óptimo y binarizar la imagen.
    - img: Imagen en escala de grises (numpy array).
    """
    
    # Calculamos histograma de la imagen
    histograma = np.zeros(256, dtype=int)
    for pixel in img.ravel():
        histograma[pixel] += 1
    
    # Normalizamos el histograma (convertimos las frecuencias en probabilidades)
    total_pixels = img.size
    probabilidad = histograma / total_pixels
    
    # Inicializamos variables para el cálculo de Otsu
    mejor_umbral = 0
    varianza_maxima = 0
    
    # Valores acumulados
    peso_fondo = 0
    media_fondo = 0
    peso_objeto = 0
    media_objeto = 0

    # Representa la intensidad promedio de la imagen
    media_total = np.sum(np.arange(256) * probabilidad)
    
    # Iteramos sobre todos los posibles umbrales
    # Para cada umbral t, dividimos los píxeles en fondo (0 a t) y objeto (t+1 a 255).
    for t in range(256):
        #  peso_fondo y peso_objeto indican qué fracción de la imagen es fondo/objeto
        peso_fondo += probabilidad[t]
        if peso_fondo == 0:
            continue
        peso_objeto = 1 - peso_fondo
        if peso_objeto == 0:
            break
        
        media_fondo += t * probabilidad[t]
        media_objeto = media_total - media_fondo
        media_fondo_norm = media_fondo / peso_fondo
        media_objeto_norm = media_objeto / peso_objeto
        
        # Calculamos la varianza entre clases
        varianza_entre_clases = peso_fondo * peso_objeto * (media_fondo_norm - media_objeto_norm) ** 2
        
        # Guardar el umbral con la mayor varianza entre clases
        if varianza_entre_clases > varianza_maxima:
            varianza_maxima = varianza_entre_clases
            mejor_umbral = t
    
    # Aplicar umbralización
    binarizada = np.where(img < mejor_umbral, 0, 255).astype(np.uint8)
    if verbose:
        print(f'El umbral fue {mejor_umbral}.')
    return binarizada # Devuelve una imagen