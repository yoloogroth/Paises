import numpy as np
import os
import pandas as pd

import tensorflow as tf

# Limpia cualquier modelo o capa previamente definida en TensorFlow
tf.keras.backend.clear_session()

# Define un modelo de red neuronal con capas densas
modelo_lineal_nuevo = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Dense_2_4'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Dense_4_8'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Dense_8_1'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Salida')
])

# Compila el modelo especificando el optimizador, la función de pérdida y las métricas
modelo_lineal_nuevo.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

# Imprime un resumen del modelo, mostrando la arquitectura y el número de parámetros
print(modelo_lineal_nuevo.summary())

# Función para generar datos en forma de círculo alrededor de un punto central
def generar_datos_circulo_nuevo(num_datos=500000, R=1, centro_lat=0, centro_lon=0):
    pi = np.pi
    # Genera ángulos aleatorios uniformemente distribuidos
    theta = np.random.uniform(0, 2 * pi, size=num_datos)

    # Genera valores positivos para el radio utilizando una distribución normal
    r_positivo = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    # Calcula las coordenadas x e y en base a coordenadas polares
    x = np.cos(theta) * r_positivo + centro_lon
    y = np.sin(theta) * r_positivo + centro_lat

    # Ajusta la precisión de las coordenadas
    x = np.round(x, 6)
    y = np.round(y, 6)

    # Crea un DataFrame con las coordenadas
    df_nuevo = pd.DataFrame({'latitud': y, 'longitud': x})
    return df_nuevo

# Genera datos en forma de círculo alrededor de Bolivia y Alemania
datos_bolivia_nuevo = generar_datos_circulo_nuevo(num_datos=100, R=2, centro_lat=-16.5000, centro_lon=-64.6667)
datos_alemania_nuevo = generar_datos_circulo_nuevo(num_datos=100, R=0.5, centro_lat=51.1657, centro_lon=10.4515)

# Combina los datos de Bolivia y Alemania en un solo conjunto de datos
X_combinado_nuevo = np.concatenate([datos_bolivia_nuevo, datos_alemania_nuevo])
X_combinado_nuevo = np.round(X_combinado_nuevo, 6)
y_combinado_nuevo = np.concatenate([np.zeros(100), np.ones(100), np.ones(100)])  # Asigna etiquetas (0 para datos circulares, 1 para Bolivia y Alemania)

# Divide el conjunto de datos en entrenamiento, prueba y validación
entrenamiento_fin_nuevo = int(0.6 * len(X_combinado_nuevo))
prueba_inicio_nuevo = int(0.8 * len(X_combinado_nuevo))
X_entrenamiento_nuevo, y_entrenamiento_nuevo = X_combinado_nuevo[:entrenamiento_fin_nuevo], y_combinado_nuevo[:entrenamiento_fin_nuevo]
X_prueba_nuevo, y_prueba_nuevo = X_combinado_nuevo[prueba_inicio_nuevo:], y_combinado_nuevo[prueba_inicio_nuevo:]
X_validacion_nuevo, y_validacion_nuevo = X_combinado_nuevo[entrenamiento_fin_nuevo:prueba_inicio_nuevo], y_combinado_nuevo[entrenamiento_fin_nuevo:prueba_inicio_nuevo]

# Entrena el modelo con los conjuntos de entrenamiento y validación durante 300 épocas
modelo_lineal_nuevo.fit(X_entrenamiento_nuevo, y_entrenamiento_nuevo, validation_data=(X_validacion_nuevo, y_validacion_nuevo), epochs=300)

# Guarda el modelo entrenado en el directorio 'linear-model/1/'
directorio_exportacion_nuevo = 'linear-model/1'  # Cambia el número del modelo si es necesario
tf.saved_model.save(modelo_lineal_nuevo, os.path.join('./', directorio_exportacion_nuevo))

# Puntos GPS para Alemania y Bolivia
puntos_gps_alemania_nuevo = [[10.4515, 51.1657], [10.0, 51.3], [10.8, 50.9], [10.6, 51.5], [10.2, 50.9]]
puntos_gps_bolivia_nuevo = [[-64.6667, -16.5000], [-65.0, -16.4], [-64.8, -16.6], [-65.1, -16.3], [-64.7, -16.5]]

# Extrae predicciones para Alemania y Bolivia
predicciones_alemania_nuevo = modelo_lineal_nuevo.predict(puntos_gps_alemania_nuevo).tolist()
predicciones_bolivia_nuevo = modelo_lineal_nuevo.predict(puntos_gps_bolivia_nuevo).tolist()

# Imprime las predicciones
print("\nPredicciones para Alemania:")
print(predicciones_alemania_nuevo)

print("\nPredicciones para Bolivia:")
print(predicciones_bolivia_nuevo)