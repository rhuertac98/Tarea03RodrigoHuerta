#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import joblib
import numpy as np
import argparse



# Cargar el modelo entrenado
def cargar_modelo(modelo_path):
    try:
        modelo = joblib.load(modelo_path)
        return modelo
    except FileNotFoundError:
        print(f"El archivo de modelo {modelo_path} no se encuentra.")
        raise

# Cargar los datos
def cargar_datos(directorio_entrada):
    # Lee tu archivo que vas a predecir
    archivo_test = directorio_entrada
    df_test = pd.read_csv(archivo_test)
    
    return df_test


# Realizar las predicciones
def realizar_predicciones(modelo, datos):
    predicciones = modelo.predict(datos)  # Supone que el modelo tiene un método predict
    return predicciones

# Guardar las predicciones
def guardar_predicciones(predicciones, directorio_salida):
    predicciones_df = pd.DataFrame(predicciones, columns=['predicciones'])
    predicciones_df.to_csv(os.path.join(directorio_salida, 'predicciones.csv'), index=False)
    print(f"Predicciones guardadas en {directorio_salida}/predicciones.csv")

def main():
    
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrenar un modelo con un archivo CSV.")
    parser.add_argument("datos", type=str, help="Ruta al archivo CSV con los datos, sólo debe tener la info. de la variables predictoras.")
    parser.add_argument("modelo", type=str, help="Ruta donde está el modelo.")
    parser.add_argument("predicciones", type=str, help="Ruta donde se guardarán las predicciones.")
    args = parser.parse_args()
  
    # Cargar el modelo
    modelo = cargar_modelo(args.modelo)
    
    # Cargar los datos
    datos = cargar_datos(args.datos)
    
    # Realizar predicciones
    predicciones = realizar_predicciones(modelo, datos)
    
    # Guardar las predicciones
    guardar_predicciones(predicciones, args.predicciones)

if __name__ == '__main__':
    main()

