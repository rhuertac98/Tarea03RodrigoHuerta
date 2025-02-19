#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Librerías y funciones

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# In[5]:


# Cargar el archivo csv en un DataFrame

archivo = 'C:/Users/rhuer/OneDrive/Escritorio/MaestríaITAM/02_Primavera_2025/ProductosDeDatos/T03/train.csv'
df_train = pd.read_csv(archivo)
df_train.head()


# In[82]:


# Separamos variables dependientes y variable objetivo

X = df_train.drop(columns=['SalePrice','Id'])  # Variables independientes
y = df_train['SalePrice']  # Variable objetivo
y_train_df = pd.DataFrame(y)


# In[7]:


X.head()


# In[90]:


y_train_df.head()


# In[94]:


sns.boxplot(x=y_train_df['SalePrice'])

# Añadir título y etiquetas (opcional)
plt.title('Precio de Venta')
plt.xlabel('Dólares')

# Mostrar la gráfica
plt.show()


# In[11]:


conteo_tipo_variables = X.dtypes.value_counts()
conteo_tipo_variables


# In[13]:


# Resumen estadístico de todas las columnas numéricas
resumen_estadistico = X.describe()
#resumen_estadistico


# In[15]:


# Resumen de columnas no numéricas
resumen_no_numerico = X.describe(include='object')
resumen_no_numerico


# In[17]:


# Correlación de variables numéricas

df_numerico = X.select_dtypes(include=['number'])
correlacion = df_numerico.corr()
#correlacion


# In[19]:


X.head(10)


# In[45]:


#Tranformación de variables categoricas con One Hot Encoder
cat_cols = X.select_dtypes(include=['object']).columns ###Generamos una "lista" de las columnas object
encoder = OneHotEncoder(sparse_output=False, drop='first') ###Generamos el encoder
encoded_data = encoder.fit_transform(X[cat_cols]) ###Aplicamos el encoder a la "lista" de columnas object. Esto es un array
encoded_dataframe = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols)) ###Transformamos el array en un data frame
X_train_OHE = pd.concat([X.drop(columns=cat_cols), encoded_dataframe], axis=1) ###Unimos con las columnas numéricas


# In[47]:


X_train_OHE.head(10)


# In[55]:


#Entrenamiento del modelo
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_OHE, y)


# In[57]:


archivo_test = 'C:/Users/rhuer/OneDrive/Escritorio/MaestríaITAM/02_Primavera_2025/ProductosDeDatos/T03/test.csv'
df_test = pd.read_csv(archivo)


# In[59]:


df_test.head(10)


# In[61]:


# Separamos variables dependientes y variable objetivo

X_test = df_test.drop(columns=['SalePrice','Id'])  # Variables independientes
y_test = df_test['SalePrice']  # Variable objetivo


# In[63]:


#Tranformación de variables categoricas con One Hot Encoder. Aplicamos el mismo encoder
encoded_data_test = encoder.fit_transform(X_test[cat_cols]) ###Aplicamos el encoder a la "lista" de columnas object. Esto es un array
encoded_dataframe_test = pd.DataFrame(encoded_data_test, columns=encoder.get_feature_names_out(cat_cols)) ###Transformamos el array en un data frame
X_test_OHE = pd.concat([X_test.drop(columns=cat_cols), encoded_dataframe_test], axis=1) ###Unimos con las columnas numéricas


# In[65]:


X_test_OHE.head()


# In[67]:


y_pred = model.predict(X_test_OHE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# In[71]:


print(f'RMSE: {rmse:.2f}')


# In[ ]:





# In[ ]:




