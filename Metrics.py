import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
from sklearn.metrics import auc
import os


def load_image(image_path):
    '''
    Carga la imagen en escala de grises.
    '''
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

def Precision(img_array, gt_img):

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)

    return TP/(TP+FP)

def CofDice(img_array,gt_img):

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    TN = np.sum((img_array == 255) & (gt_img == 255))  # 255 en ambos (blanco)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((img_array == 255) & (gt_img == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    return 2*TP/(2*TP + FP + FN)

def IndJaccard(img_array,gt_img):

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    TN = np.sum((img_array == 255) & (gt_img == 255))  # 255 en ambos (blanco)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((img_array == 255) & (gt_img == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    return TP/(TP + FP + FN)

def Sensitivity(img_array,gt_img):

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    TN = np.sum((img_array == 255) & (gt_img == 255))  # 255 en ambos (blanco)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((img_array == 255) & (gt_img == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    return TP / (TP + FN) 

def Specificity(img_array, gt_img):

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    TN = np.sum((img_array == 255) & (gt_img == 255))  # 255 en ambos (blanco)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((img_array == 255) & (gt_img == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    return TN / (TN + FP) 

def Accuracy(img_array, gt_img):
    TP = np.sum((img_array == 0) & (gt_img == 0))  # 0 en ambos (negro)
    TN = np.sum((img_array == 255) & (gt_img == 255))  # 255 en ambos (blanco)
    FP = np.sum((img_array == 0) & (gt_img == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((img_array == 255) & (gt_img == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    return (TP + TN)/(TP + TN + FP + FN)


def metricas_PostProcess(img_array, gt_img):
    print("Se obtuvieron los siguientes resultados: \n")
    print(f'    Accuracy (ACC):     {Accuracy(img_array, gt_img)}.')
    print(f'    Sensitivity (TPR):  {Sensitivity(img_array, gt_img)}.')
    print(f'    Specificity (SPC):  {Specificity(img_array, gt_img)}.')
    print(f'    Precision (PVV):    {Accuracy(img_array, gt_img)}.')
    print(f'    Dice Coef (DSC):    {CofDice(img_array, gt_img)}.')
    print(f'    Ind Jaccard (IoU):  {IndJaccard(img_array, gt_img)}.')
    




def tptn(umbral, img_array, gtim):

    '''
    Dado un umbral, una imagen (en np array) y el ground truth
    Binarizamos la imagen a ese umbral
    Comparamos con sensitivity, specificity que tanto se parece
    al gt (ground truth)
    '''

    binarizada =  np.where(img_array < umbral, 0, 255)

    # Cálculo de TP, TN, FP, FN
    TP = np.sum((binarizada == 0) & (gtim == 0))  # 0 en ambos (negro)
    TN = np.sum((binarizada == 255) & (gtim == 255))  # 255 en ambos (blanco)
    FP = np.sum((binarizada == 0) & (gtim == 255))  # 0 en binarizada, 255 en GT (falso positivo)
    FN = np.sum((binarizada == 255) & (gtim == 0))  # 255 en binarizada, 0 en GT (falso negativo)

    # Cálculo de Sensibilidad y Especificidad
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return sensitivity, specificity

def ROC_value(TopHat, gt_array):

    sensitivity = np.zeros(255)
    specificity = np.zeros(255)

    for i in range(0, 255):
        sen, spe = tptn(i, TopHat, gt_array)
        sensitivity[i] = sen
        specificity[i] = spe

    # Calculamos 1 - especificidad
    one_minus_spe = 1 - specificity
    umbrales = np.arange(0, len(specificity))

    auc_score = auc(one_minus_spe, sensitivity)

    return auc_score


def curva_roc(sen, spe):
    '''
    Genera y muestra la curva ROC a partir de sensibilidad y especificidad.

    Parámetros:
    sen : array-like  -> Sensibilidad (TPR).
    spe : array-like  -> Especificidad (TNR).

    - Calcula 1 - especificidad para el eje X.
    - Grafica la curva ROC con `plotly.graph_objects`.
    - Muestra el área bajo la curva (AUC).
    '''



    # Calculamos 1 - especificidad
    one_minus_spe = 1 - spe
    umbrales = np.arange(0, len(sen))


    # Crear el gráfico de la curva ROC
    fig = go.Figure()

    hover_text = [f"Umbral {u}: ({round(x, 3)}, {round(y, 3)})" 
                  for u, x, y in zip(umbrales, one_minus_spe, sen)]


    # Añadir la curva ROC
    fig.add_trace(go.Scatter(
        x=one_minus_spe,  # Eje X: 1 - especificidad
        y=sen,  # Eje Y: sensibilidad
        mode='lines',  # Graficar línea
        name='Curva ROC',
        line=dict(color='blue'),
        hovertext = hover_text,
        hoverinfo="text"
    ))

    # Añadir línea diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1],  # Desde (0,0) hasta (1,1)
        y=[0, 1],  # Desde (0,0) hasta (1,1)
        mode='lines',  # Graficar línea
        name='Aleatorio',
        line=dict(dash='dash', color='gray')
    ))

    # Configurar el layout
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='1 - Especificidad',
        yaxis_title='Sensibilidad',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        width=600,  
        height=500
    )

    auc_score = auc(one_minus_spe, sen)

    print(f"El área bajo la curva ROC (AUC) es: {auc_score}")

    fig.show()
    