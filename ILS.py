import numpy as np
import TopHat as th
import Metrics as roc
import matplotlib.pyplot as plt

def perturb(kernel):
    '''
    Perturba el kernel agregando pequeñas variaciones.
    '''

    perturbed_kernel = kernel.copy()
    perturbation = np.random.choice([0, 1], size=kernel.shape, p=[0.90, 0.10])  # 10% de cambio
    perturbed_kernel = np.logical_xor(perturbed_kernel, perturbation)  # Invierte algunos valores
    
    return perturbed_kernel.astype(np.uint8)

def perturb_small(kernel):
    '''
    Perturba el kernel agregando pequeñas variaciones para la busqueda local.
    '''

    perturbed_kernel = kernel.copy()
    perturbation = np.random.choice([0, 1], size=kernel.shape, p=[0.97, 0.03])  # 5% de cambio
    perturbed_kernel = np.logical_xor(perturbed_kernel, perturbation)  # Invierte algunos valores
    
    return perturbed_kernel.astype(np.uint8)


def local_search(kernel, img_array, gt_array, max_local_steps=10):
    '''
    Aplica el Top-Hat transform y evalúa la curva ROC para refinar el kernel.
    Mientras que busca el mejor kernel cercano
    '''

    kernel_best = kernel
    filtered_images = th.BlackHat(img_array, kernel_best)  
    Az_best = roc.ROC_value(filtered_images, gt_array)

    for _ in range(max_local_steps):
        kernel_candidate = perturb_small(kernel_best)  # Genera una variación muy pequena
        Az_candidate = roc.ROC_value(th.BlackHat(img_array, kernel_candidate), gt_array)  # Evalúa

        if Az_candidate > Az_best:  # Si mejora, se actualiza
            kernel_best, Az_best = kernel_candidate, Az_candidate

    return kernel_best, Az_best

def accept(kernel_current, kernel_new, Az_current, Az_new):
    '''
    Acepta el nuevo kernel si su valor Az es mayor.
    '''

    return (kernel_new, Az_new) if Az_new > Az_current else (kernel_current, Az_current)

def iterated_local_search(initial_kernel, img_array, gt_array, max_iter=50):
    '''
    Iterated Local Search.
    '''

    kernel = initial_kernel.copy()
    kernel_best, Az_best = local_search(kernel, img_array, gt_array)

    for i in range(max_iter):
        kernel_star = perturb(kernel_best)  # Perturbación aleatoria
        plt.figure(figsize=(3,3))
        plt.imshow(kernel_star, cmap='gray')
        plt.axis('off')
        plt.show()
        kernel_t_prime, Az_t_prime = local_search(kernel_star, img_array, gt_array)  # Aplicación de búsqueda local
        kernel_best, Az_best = accept(kernel_best, kernel_t_prime, Az_best, Az_t_prime)  # Criterio de aceptación
        print(f'En interacion {i} el mejor accuracy Az es {Az_best}')

    return kernel_best, Az_best