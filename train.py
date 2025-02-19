#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import argparse

def cargar_datos(archivo_csv):
    """Cargar datos desde un archivo CSV."""
    return pd.read_csv(archivo_csv)

def entrenar_modelo(X_train, y_train):
    """Entrenar un modelo de regresión logística."""
    modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def guardar_modelo(modelo, nombre_archivo):
    """Guardar el modelo entrenado en un archivo."""
    joblib.dump(modelo, nombre_archivo)

def main():
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrenar un modelo con un archivo CSV.")
    parser.add_argument("archivo_csv", type=str, help="Ruta al archivo CSV con los datos. La última columna deber ser la variable objetivo.")
    parser.add_argument("modelo_salida", type=str, help="Ruta donde se guardará el modelo entrenado.")
    args = parser.parse_args()

    # Cargar datos
    datos = cargar_datos(args.archivo_csv)
    
    # Separar las características (X) y la etiqueta (y)
    # Suponiendo que la última columna es la etiqueta
    X_train = datos.iloc[:, :-1]  # Todas las columnas excepto la última
    y_train = datos.iloc[:, -1]   # Última columna


    # Entrenar el modelo
    modelo = entrenar_modelo(X_train, y_train)

    # Guardar el modelo entrenado
    guardar_modelo(modelo, args.modelo_salida)
    print(f"Modelo guardado en: {args.modelo_salida}")

if __name__ == "__main__":
    main()

