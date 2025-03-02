import numpy as np
from PIL import Image
import os
import cv2
import TopHat as th
import otsu as ot
import ComponentesConexos as cc

def concatenate_images_from_folder(folder_path, axis=1):
    '''
    Carga y concatena todas las imágenes en una carpeta

    Parámetros:
    - folder_path: Ruta de la carpeta que contiene las imágenes.
    - axis: 0 para concatenar verticalmente, 1 para horizontalmente.

    Retorna:
    - Imagen concatenada como un `np.array` o lanza un error si no hay imágenes válidas.
    '''
    # Obtener lista de archivos en la carpeta
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'gif', 'pgm'))])
    if not image_files:
        raise ValueError("No se encontraron imágenes en la carpeta.")

    images = []

    for file in image_files:
        img_path = os.path.join(folder_path, file)

        img = Image.open(img_path).convert("RGB")  # Convertir a RGB para evitar problemas
        img = np.array(img)  # Convertir a numpy array
        
        images.append(img)

    # Filtrar imágenes que no se cargaron correctamente
    if not images:
        raise ValueError("No se pudieron cargar imágenes válidas.")

    # Convertir todas a la misma altura/ancho según el `axis`
    if axis == 1:  # Concatenar horizontalmente
        height = min(img.shape[0] for img in images)
        images = [np.array(Image.fromarray(img).resize((int(img.shape[1] * (height / img.shape[0])), height))) for img in images]
    else:  # Concatenar verticalmente
        width = min(img.shape[1] for img in images)
        images = [np.array(Image.fromarray(img).resize((width, int(img.shape[0] * (width / img.shape[1]))))) for img in images]

    concatenated_image = np.concatenate(images, axis=axis)
    
    return concatenated_image


def process_and_save_images(input_folder, output_folder, kernel):
    '''
    Aplica la función `TopHat` a todas las imágenes en `input_folder`,
    y guarda los resultados en `output_folder`.

    Parámetros:
    - input_folder: Carpeta con las imágenes originales.
    - output_folder: Carpeta donde se guardarán los resultados.
    - kernel: kernel para la función `TopHat`.

    
    Guarda las imágenes procesadas en `output_folder`.
    '''

    # Asegurar que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de imágenes en la carpeta de entrada
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'pgm'))])


    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)

        # Cargar la imagen original
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"Error cargando imagen: {img_path}")
            continue


        # Aplicar la función
        resultado = th.BlackHat(original, kernel)

        # Verificar que resultado sea un array válido
        if not isinstance(resultado, np.ndarray):
            print(f"Error en el procesamiento de {img_file}")
            continue

        # Construir la ruta de salida
        output_path = os.path.join(output_folder, img_file)

        # Guardar la imagen procesada
        cv2.imwrite(output_path, resultado)

    print("Proceso completado.")



def process_and_save_images_binar(input_folder, output_folder):
    '''
    Aplica la función de binarizacion a todas las imágenes en `input_folder`,
    y guarda los resultados en `output_folder`.

    Parámetros:
    - input_folder: Carpeta con las imágenes originales.
    - output_folder: Carpeta donde se guardarán los resultados.
    
    Guarda las imágenes procesadas en `output_folder`.
    '''

    # Asegurar que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de imágenes en la carpeta de entrada
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'pgm'))])


    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)

        # Cargar la imagen original
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"Error cargando imagen: {img_path}")
            continue


        # Aplicar la función
        resultado = ot.otsu_threshold(original, verbose=0)

        # Verificar que resultado sea un array válido
        if not isinstance(resultado, np.ndarray):
            print(f"Error en el procesamiento de {img_file}")
            continue

        # Construir la ruta de salida
        output_path = os.path.join(output_folder, img_file)

        # Guardar la imagen procesada
        cv2.imwrite(output_path, resultado)

    print("Proceso completado.")




def process_and_save_images_conected(input_folder, output_folder):
    '''
    Aplica la función de componentes conexos a todas las imágenes en `input_folder`,
    y guarda los resultados en `output_folder`.

    Parámetros:
    - input_folder: Carpeta con las imágenes originales.
    - output_folder: Carpeta donde se guardarán los resultados.
    
    Guarda las imágenes procesadas en `output_folder`.
    '''

    # Asegurar que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de imágenes en la carpeta de entrada
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'pgm'))])


    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)

        # Cargar la imagen original
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"Error cargando imagen: {img_path}")
            continue


        # Aplicar la función
        resultado = cc.Conectivity(original)

        # Verificar que resultado sea un array válido
        if not isinstance(resultado, np.ndarray):
            print(f"Error en el procesamiento de {img_file}")
            continue

        # Construir la ruta de salida
        output_path = os.path.join(output_folder, img_file)

        # Guardar la imagen procesada
        cv2.imwrite(output_path, resultado)

    print("Proceso completado.")
