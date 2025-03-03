# Evaluación en Segmentación de Vasos Sanguíneos

Este proyecto tiene como objetivo realizar una segmentación de vasos sanguíneos en imágenes biomédicas de DRIVE. 

## Estructura del Proyecto

El proyecto está organizado en varios archivos de prueba en formato `.ipynb`, que documentan el proceso de evaluación paso a paso. Además, se incluyen librerías de funciones auxiliares en archivos `.py`.

### Archivos y Directorios Principales

- **Notebooks de prueba (`.ipynb`)**: Cada paso del proceso tiene su propio archivo de prueba, donde se documenta el desarrollo y evaluación de los métodos.
- **Librerías auxiliares (`.py`)**: Archivos que contienen funciones reutilizables para distintas etapas de la evaluación.
- **Datos**: Carpeta donde se almacenan las imágenes biomédicas utilizadas para la segmentación. (`Drive` y `DB_aNGIOGRAMS_134`)

## Pasos Implementados

1. **Preprocesamiento**: Mejora de la calidad de las imágenes antes de la segmentación. (No fue usado para las metricas)
2. **Segmentación Inicial**: Aplicación de técnicas de segmentación básicas. (Top-hat)
3. **Postprocesamiento**: Refinamiento de los resultados obtenidos. (Binarizacion y Componentes Conexos)
4. **Métricas de Evaluación**: Cálculo de métricas para comparar la calidad de los métodos implementados. 
5. **ILS (Inpainting Learning Strategy) [En Desarrollo]**: Método en proceso de implementación para mejorar la regilla.
