'''
------------------------------------------------
----------  MODELO FINAL COMPETICIÓN  ----------
------------------------------------------------
'''

# Importaciones necesarias.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
from rich import print


# Cargamos los datos de entrenamiento y competición.

# Datos disponibles.
disp_df = pd.read_csv("disp_st13ns1.txt.bz2",
                      compression="bz2",
                      index_col=0)

# Datos competición.
comp_df = pd.read_csv("comp_st13ns1.txt.bz2",
                      compression="bz2",
                      index_col=0)

# Semilla para la reproducibilidad.
np.random.seed(13)

# Datos.
X = disp_df.drop('salida', axis=1)

# Etiquetas.
y = disp_df.salida

# Queremos entrenar el modelo con todos los datosdisponibles y predecir los datos de competición.
X_train = X
y_train = y
X_test = comp_df

# Normalizamos los datos.
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_n = scaler.transform(X_train)
X_test_n = scaler.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(y_train.values.reshape(-1, 1))
y_train_n = scaler.transform(y_train.values.reshape(-1, 1))


# Entrenamos el modelo y hacemos las predicciones sobre los datos de competición.
model = LinearRegression()

model.fit(X_train_n, y_train_n)

# Guardamos el modelo entrenado final en el fichero modelo_final.pkl
with open("modelo_final.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred_n = model.predict(X_test_n)

y_pred = scaler.inverse_transform(y_pred_n)

# Guardamos las predicciones en el fichero predicciones.scv.
pd.DataFrame(y_pred, index=X_test.index, columns=["salida"]).to_csv("predicciones.csv")
