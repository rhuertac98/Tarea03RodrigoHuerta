#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import os
import argparse
from sklearn.preprocessing import OneHotEncoder


# In[3]:


directorio_datos = 'data/'


# In[7]:


# Lee los datos desde el directorio 'data/'
def carga_datos(nombre_datos):
    datos_raw = os.path.join(directorio_datos, nombre_datos)  # Ajusta el nombre del archivo
    if os.path.exists(datos_raw):
        data = pd.read_csv(datos_raw)
        return data
    else:
        raise FileNotFoundError(f"El archivo {datos_raw} no existe.")


# In[11]:


def OHE_datos(data):
    cat_cols = data.select_dtypes(include=['object']).columns ###Generamos una "lista" de las columnas object
    encoder = OneHotEncoder(sparse_output=False, drop='first') ###Generamos el encoder
    encoded_data = encoder.fit_transform(data[cat_cols]) ###Aplicamos el encoder a la "lista" de columnas object. Esto es un array
    encoded_dataframe = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols)) ###Transformamos el array en un data frame
    data_OHE = pd.concat([data.drop(columns=cat_cols), encoded_dataframe], axis=1)
    
    return data_OHE


# In[13]:


def guardar_datos(data):
    archivo_OHE = os.path.join(directorio_datos, 'datos_OHE.csv')
    data.to_csv(archivo_OHE, index=False)
    print(f"Datos codificados con OHE en {archivo_OHE}")


# In[21]:


def main():
    try:
        parser = argparse.ArgumentParser(description='Procesar archivos CSV de datos.')
        parser.add_argument('nombre_archivo', type=str, help='Nombre del archivo CSV para aplicar ONE HOT ENCODING')
        args = parser.parse_args()

        nombre_datos = args.nombre_archivo
        print(f"Procesando el archivo: {nombre_datos}")
        
        # Cargar datos
        datos = carga_datos(nombre_datos)
        print("Datos cargados exitosamente.")
        
        # Procesar los datos
        OHE_data = OHE_datos(datos)
        print("Datos procesados.")
        
        # Guardar los datos procesados
        guardar_datos(OHE_data)
    except Exception as e:
        print(f"Hubo un error: {e}")


# In[27]:


if __name__ == '__main__':
    main()


# In[ ]:




