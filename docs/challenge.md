# Software Engineer (ML & LLMs) Challenge

## Overview

Este documento detalla el trabajo realizado para el desafío de **Software Engineer (ML & LLMs)**. Se ha implementado un modelo de predicción de retrasos en vuelos utilizando el conjunto de datos proporcionado. A continuación, se describen las partes del proyecto y las decisiones tomadas durante el desarrollo.

## Parte I: Modelado

### Clase DelayModel

Se ha implementado la clase `DelayModel`, que contiene todos los métodos necesarios para cargar, procesar datos, entrenar el modelo y guardar el modelo entrenado. Aquí hay un desglose de los métodos:

1. **load_data(file_path)**: 
   - Carga los datos desde un archivo CSV utilizando `pandas`.

2. **preprocess_data(data)**: 
   - Limpia y prepara los datos, incluyendo:
     - Conversión de columnas de fecha.
     - Creación de nuevas columnas: `high_season`, `min_diff`, `period_day` y `delay`.
     - Manejo de columnas faltantes.

3. **train(file_path)**: 
   - Carga y procesa los datos, define las características y el objetivo, divide el conjunto de datos en entrenamiento y prueba, y entrena un modelo de `RandomForestClassifier`.
   - Evalúa el modelo y muestra la precisión, matriz de confusión y un informe de clasificación.

4. **save_model(file_path)**: 
   - Guarda el modelo entrenado en un archivo utilizando `pickle`.

### Resultados

El modelo entrenado alcanzó una precisión del 100% (reemplaza con el valor real). Se generó una matriz de confusión y un informe de clasificación que muestra el desempeño del modelo en la predicción de retrasos.

## Parte II: API con FastAPI

Se creó una API utilizando `FastAPI` en el archivo `api.py`. La API expone un endpoint que permite a los usuarios enviar solicitudes de predicción de retrasos. A continuación se presentan los detalles de la implementación:

- **Endpoints**: Se definió un endpoint `/predict` que acepta parámetros necesarios para realizar la predicción.
- **Modelo**: El modelo entrenado es cargado y utilizado para predecir la probabilidad de retraso basado en las características de entrada.


## Parte III: Despliegue en Google Cloud

La API se desplegó en Google Cloud utilizando Cloud Run, pero presento problemas para levantar servicio, solicitaba agregar tarjeta de credito

## Parte IV: Implementación de CI/CD
Se completaron los archivos ci.yml y cd.yml para la integración y entrega continua del proyecto. Estos archivos están configurados para ejecutar pruebas automatizadas y desplegar la aplicación de acuerdo con las mejores prácticas de CI/CD.

Sin embargo, se debe mencionar que la implementación del despliegue en Google Cloud Platform (GCP) no se completó en esta etapa. A pesar de tener configuradas las acciones de CI/CD, el paso final de despliegue en GCP requiere una configuración adicional quedo pendiente.

