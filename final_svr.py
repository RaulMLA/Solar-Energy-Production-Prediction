'''
------------------------------------------------
----------  MODELO FINAL COMPETICIÓN  ----------
------------------------------------------------
'''

# Importaciones necesarias.
import numpy as np
import pandas as pd
from sklearn.svm import SVR
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

variables_meteorologicas = ['apcp_sf', 'dlwrf_s', 'dswrf_s', 'pres_msl', 'pwat_eatm', 'spfh_2m', 'tcdc_eatm', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m', 'tmp_sfc', 'ulwrf_sfc', 'ulwrf_tatm', 'uswrf_sfc']


mean_df = disp_df.iloc[:, :-1].groupby(np.arange(len(disp_df.columns)-1)//5, axis=1).mean()
mean_df.columns = [f'{name}_media' for name in variables_meteorologicas]

mean_df_comp = comp_df.iloc[:, :-1].groupby(np.arange(len(comp_df.columns)-1)//5, axis=1).mean()
mean_df_comp.columns = [f'{name}_media' for name in variables_meteorologicas]

mean_df['salida'] = disp_df['salida']

df_reducida = mean_df.drop(['apcp_sf_media', 'pres_msl_media', 'tcdc_eatm_media', 'tcolc_eatm_media'], axis=1)

df_reducida_comp = mean_df_comp.drop(['apcp_sf_media', 'pres_msl_media', 'tcdc_eatm_media', 'tcolc_eatm_media'], axis=1)

# mostramos los nombres de las columnas.
print(df_reducida.columns)
print(df_reducida_comp.columns)

# Datos.
X = df_reducida.drop('salida', axis=1)

# Etiquetas.
y = df_reducida.salida

# Queremos entrenar el modelo con todos los datosdisponibles y predecir los datos de competición.
X_train = X
y_train = y
X_test = df_reducida_comp

# Normalizamos los datos.
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_n = scaler.transform(X_train)
X_test_n = scaler.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(y_train.values.reshape(-1, 1))
y_train_n = scaler.transform(y_train.values.reshape(-1, 1))


# Entrenamos el modelo y hacemos las predicciones sobre los datos de competición.
#'C': 2, 'coef0': 1, 'degree': 2, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'poly', 'tol': 0.0001
model = SVR(C=2, coef0=1, degree=2, epsilon=0.1, gamma='scale', kernel='poly', tol=0.0001)

model.fit(X_train_n, y_train_n.ravel())

# Guardamos el modelo entrenado final en el fichero modelo_final.pkl
with open("modelo_final.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred_n = model.predict(X_test_n)

y_pred = scaler.inverse_transform(y_pred_n.reshape(-1, 1))

# Guardamos las predicciones en el fichero predicciones.scv.
pd.DataFrame(y_pred, index=X_test.index, columns=["salida"]).to_csv("predicciones.csv")
