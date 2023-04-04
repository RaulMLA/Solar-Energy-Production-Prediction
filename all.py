#!pip install --upgrade linear-tree
#!pip install statsmodels

# Importación de datos.
import numpy as np
import pandas as pd

# EDA
import seaborn as sns
import matplotlib.pyplot as plt

# División de los datos en entrenamiento y test.
from sklearn.model_selection import train_test_split

# Evaluación de modelos sin ajuste de hiperparámetros.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import time

from sklearn.model_selection import PredefinedSplit, cross_val_score

from sklearn.dummy import DummyRegressor

# Evaluación de modelos con ajuste de hiperparámetros.
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Métodos avanzados.
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Prints de colores.
from rich import print



#------------------------------------------------------------
'''Funciones importantes.'''
#------------------------------------------------------------

def mse (y_test, y_test_pred):
    '''Mean squared error.'''
    return metrics.mean_squared_error(y_test, y_test_pred)

def rmse (y_test, y_test_pred):
    '''Root mean squared error.'''
    return np.sqrt(mse(y_test, y_test_pred))

def mae (y_test, y_test_pred):
    '''Mean absolute error.'''
    return metrics.mean_absolute_error(y_test, y_test_pred)


print('[bold yellow]' + '-' * 60 + '\n' + '-' * 60 + '\n' + '-' * 60 + '\n' + '[/bold yellow]')

#------------------------------------------------------------
'''Importamos los datos.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nImportando los datos...\n' + '-' * 60 + '[/bold red]')


# Datos disponibles.
disp_df = pd.read_csv('disp_st13ns1.txt.bz2',
                      compression='bz2',
                      index_col=0)

# Datos competición.
comp_df = pd.read_csv('comp_st13ns1.txt.bz2',
                      compression='bz2',
                      index_col=0)

# Mostramos la información de cada conjunto de datos.
print(f'El conjunto de datos disponibles tiene {len(disp_df)} instancias.')
print(f'El conjunto de datos de competición tiene {len(disp_df)} instancias.')


# Datos.
X = disp_df.drop('salida', axis=1)

# Etiquetas.
y = disp_df.salida

# Semilla para reproducibilidad (grupo de laboratorio).
np.random.seed(13)

print()



#------------------------------------------------------------
'''EDA'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nAnálisis Exploratorio de Datos (EDA)\n' + '-' * 60 + '[/bold red]')

# Mostramos todos los datos.
print('Matriz de atributos:\n\n', X)
print('\n\nVector de la variable de respuesta:\n\n', y)

# Mostramos el tipo de dato de una variable meteorológica y de un valor de la variable de respuesta.
print('\nEjemplo de tipo de dato de variable meteorológica:', type(X['apcp_sf1_1'][0]))
print('Ejemplo de tipo de dato de variable de respuesta:', type(y[0]))

# Contamos el número de missing values.
print('\nMissing values: ', disp_df.isnull().values.sum())

# Media de cada variable meteorológica
variables_meteorologicas = ['apcp_sf', 'dlwrf_s', 'dswrf_s', 'pres_msl', 'pwat_eatm', 'spfh_2m', 'tcdc_eatm', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m', 'tmp_sfc', 'ulwrf_sfc', 'ulwrf_tatm', 'uswrf_sfc']

# Creamos un dataframe con la media de cada variable meteorológica.
mean_df = disp_df.iloc[:, :-1].groupby(np.arange(len(disp_df.columns)-1)//5, axis=1).mean()
mean_df.columns = [f'{name}_media' for name in variables_meteorologicas]
mean_df['salida'] = disp_df['salida']
#print(mean_df)

# Boxplot de cada variable meteorológica con seaborn.
'''
sns.set(style='ticks')

# Seleccionar solo las columnas de las variables meteorológicas
cols = mean_df.columns[:-1]

# Crear un diagrama de cajas para cada variable
for col in cols:
    sns.boxplot(data=mean_df, x=col)
    plt.show()
'''

# Correlación entre variables meteorológicas.

sns.set_theme(style='white')

# Computar la matriz de correlación de la media de las variables meteorológicas.
corr = mean_df.corr()

#print(corr)

# Máscara para el triángulo superior.
mask = np.triu(np.ones_like(corr, dtype=bool))

# Configuración de la figura matplotlib.
f, ax = plt.subplots(figsize=(15, 15))

# Generación de un mapa de colores divergentes personalizado.
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Dibujar el mapa de calor con la máscara y el mapa de colores adecuados.
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .5}, annot=True)

#plt.show()

# Guardar la grafica de correlacion como archivo .jpg.
plt.savefig('correlacion.jpg')

print()



#------------------------------------------------------------
'''División de los datos en entrenamiento y test.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nDivisión de los datos en entrenamiento y test.\n' + '-' * 60 + '[/bold red]')

# Entrenamiento (10 primeros años) y test (2 últimos años).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/12, random_state=13, shuffle=False)

# Comprobamos que los datos se hayan dividido como queremos.
print('Datos de entrenamiento:', X_train.shape, y_train.shape)   # 3650 días -> 10 años.
print('Datos de test:', X_test.shape, y_test.shape)              # 720 días  ->  2 años.

# Convertir dataframe a numpy array.
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Normalizamos los datos.
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_n = scaler.transform(X_train)
X_test_n = scaler.transform(X_test)

scaler = MinMaxScaler()
scaler.fit(y_train.reshape(-1, 1))
y_train_n = scaler.transform(y_train.reshape(-1, 1))
y_test_n = scaler.transform(y_test.reshape(-1, 1))

print()



#------------------------------------------------------------
'''Evaluación de modelos simples sin ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de modelos simples sin ajuste de hiperparámetros.\n' + '-' * 60 + '[/bold red]')

# Volvemos a dividir los datos en entrenamiento y test porque la parttición test solo la usaremos en la evaluación final.
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=3/10, random_state=13, shuffle=False)

# Dividir también los datos normalizados.
X_train_train_n, X_train_test_n, y_train_train_n, y_train_test_n = train_test_split(X_train_n, y_train_n, test_size=3/10, random_state=13, shuffle=False)

# Muestra el tamaño de los datos de entrenamiento y test nuevos.
print('Datos train_train: ' , X_train_train.shape, y_train_train.shape)   # 2550 días -> 7 años.
print('Datos train_test: ' , X_train_test.shape, y_train_test.shape)      # 1100 días  ->  3 años.


# KNN.
print('\n[bold yellow]KNN\n----[/bold yellow]')

# [KNN] MODELO BASE.
print('\n[yellow]Modelo base[/yellow]')
base_knn = KNeighborsRegressor()

# Entrenamiento (con medición del tiempo).
start = time.time()
base_knn.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_knn = end - start
print(f'Tiempo de entrenamiento: {time_knn:.5f}')

# Predicciones del conjunto de test.
y_pred_n = base_knn.predict(X_train_test_n)

# Denormalizar los datos (aunque se podría RMSE y MAE sin denormalizar).
y_pred = scaler.inverse_transform(y_pred_n)

# Cálculo del error cuadrático medio.
rmse_knn = rmse(y_train_test, y_pred)
print(f'\nRMSE: {rmse_knn}')

# Cálculo del error absoluto medio.
mae_knn = mae(y_train_test, y_pred)
print(f'MAE: {mae_knn}')


# [KNN] MODELO VALIDACIÓN CRUZADA.
print('\n[yellow]Modelo validación cruzada[/yellow]')

# Usar predefined split para la validación cruzada.

# Número de días de entrenamiento y test.
N_train = 7*365
N_test = 3*365

# Crear el selector de validación cruzada.
selector = [-1] * N_train + [0] * N_test

# Crear el objeto PredefinedSplit.
ps = PredefinedSplit(selector)

'''
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
  print(f'Size of the train set: {train.shape}')
  print(f'Size of the valid set: {valid.shape}')
  print(f'Indices of the train set: {train}')
  print(f'Indices of the valid set: {valid}')
'''

# Usar el predefined split para la validación cruzada (con medición del tiempo).
cv_knn = KNeighborsRegressor()

start = time.time()
rmse_knn_cv = cross_val_score(cv_knn, X_train_n, y_train_n, cv=ps, scoring='neg_root_mean_squared_error')
end = time.time()
time1_knn_cv = end - start

start = time.time()
mae_knn_cv = cross_val_score(cv_knn, X_train_n, y_train_n, cv=ps, scoring='neg_mean_absolute_error')
end = time.time()
time2_knn_cv = end - start

print(f'Tiempo de entrenamiento (RMSE): {time1_knn_cv:.5f}')
print(f'Tiempo de entrenamiento (MAE): {time2_knn_cv:.5f}')
print(f'\nRMSE: {-rmse_knn_cv.mean()}')
print(f'MAE: {-mae_knn_cv.mean()}')


# [KNN] MODELOS DUMMY.
print('\n[yellow]Modelos dummy[/yellow]')
dummy_1_knn = DummyRegressor(strategy='mean')
dummy_2_knn = DummyRegressor(strategy='median')

# Entrenamiento de los modelos dummy (con medición del tiempo).
start = time.time()
dummy_1_knn.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_knn_dm1 = end - start

start = time.time()
dummy_2_knn.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_knn_dm2 = end - start

print(f'Tiempo de entrenamiento (mean): {time_knn_dm1:.5f}')
print(f'Tiempo de entrenamiento (median): {time_knn_dm2:.5f}')

# Predicciones del conjunto de test.
y_pred_dummy_1_n = dummy_1_knn.predict(X_train_test_n).reshape(-1, 1)
y_pred_dummy_2_n = dummy_2_knn.predict(X_train_test_n).reshape(-1, 1)

# Denormalizar los datos (aunque se podría RMSE y MAE sin denormalizar).
y_pred_dummy_1 = scaler.inverse_transform(y_pred_dummy_1_n)
y_pred_dummy_2 = scaler.inverse_transform(y_pred_dummy_2_n)

# Cálculo del error cuadrático medio.
rmse_knn_dm1 = rmse(y_train_test, y_pred_dummy_1)
rmse_knn_dm2 = rmse(y_train_test, y_pred_dummy_2)
print(f'\nRMSE (mean): {rmse_knn_dm1}')
print(f'RMSE (median): {rmse_knn_dm2}')

# Cálculo del error absoluto medio.
mae_knn_dm1 = mae(y_train_test, y_pred_dummy_1)
mae_knn_dm2 = mae(y_train_test, y_pred_dummy_2)
print(f'MAE (mean): {mae_knn_dm1}')
print(f'MAE (median): {mae_knn_dm2}')

# Relación entre el error del modelo y el error de los modelos dummy.
print(f'\nRMSE dummy (mean)/RMSE KNN: {rmse_knn_dm1/rmse_knn}')
print(f'RMSE dummy (median)/RMSE KNN: {rmse_knn_dm2/rmse_knn}')
print(f'MAE dummy (mean)/MAE KNN: {mae_knn_dm1/mae_knn}')
print(f'MAE dummy (median)/MAE KNN: {mae_knn_dm2/mae_knn}')


# Árbol de decisión.
print('\n[bold yellow]Árbol de decisión\n------------------[/bold yellow]')

# [Árbol de decisión] MODELO BASE.
print('\n[yellow]Modelo base[/yellow]')
base_tree = DecisionTreeRegressor()

# Entrenamiento (con medición del tiempo).
start = time.time()
base_tree.fit(X_train_train, y_train_train)
end = time.time()
time_tree = end - start
print(f'Tiempo de entrenamiento: {time_tree:.5f}')

# Predicciones del conjunto de test.
y_pred = base_tree.predict(X_train_test)

# Cálculo del error cuadrático medio.
rmse_tree = rmse(y_train_test, y_pred)
print(f'\nError cuadrático medio del modelo Árbol de decisión: {rmse_tree}')

# Cálculo del error absoluto medio.
mae_tree = mae(y_train_test, y_pred)
print(f'Error absoluto medio del modelo Árbol de decisión: {mae_tree}')


# [Árbol de decisión] MODELO VALIDACIÓN CRUZADA.
print('\n[yellow]Modelo validación cruzada[/yellow]')

# Usar predefined split para la validación cruzada.

'''
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
    print(f'Size of the train set: {train.shape}')
    print(f'Size of the valid set: {valid.shape}')
    print(f'Indices of the train set: {train}')
    print(f'Indices of the valid set: {valid}')
'''

# Usar el predefined split para la validación cruzada (con medición del tiempo).
cv_tree = DecisionTreeRegressor()

start = time.time()
rmse_tree_cv = cross_val_score(cv_tree, X_train, y_train, cv=ps, scoring='neg_root_mean_squared_error')
end = time.time()
time1_tree_cv = end - start

start = time.time()
mae_tree_cv = cross_val_score(cv_tree, X_train, y_train, cv=ps, scoring='neg_mean_absolute_error')
end = time.time()
time2_tree_cv = end - start


print(f'Tiempo de entrenamiento: {time1_tree_cv:.5f}')
print(f'Tiempo de entrenamiento: {time2_tree_cv:.5f}')
print(f'\nRMSE: {-rmse_tree_cv.mean()}')
print(f'MAE: {-mae_tree_cv.mean()}')


# [Árbol de decisión] MODELOS DUMMY.
print('\n[yellow]Modelos dummy[/yellow]')
dummy_1_tree = DummyRegressor(strategy='mean')
dummy_2_tree = DummyRegressor(strategy='median')

# Entrenamiento de los modelos dummy (con medición del tiempo).
start = time.time()
dummy_1_tree.fit(X_train_train, y_train_train)
end = time.time()
time_tree_dm1 = end - start

start = time.time()
dummy_2_tree.fit(X_train_train, y_train_train)
end = time.time()
time_tree_dm2 = end - start

print(f'Tiempo de entrenamiento (mean): {time_tree_dm1:.5f}')
print(f'Tiempo de entrenamiento (median): {time_tree_dm2:.5f}')

# Predicciones del conjunto de test.
y_pred_dummy_1 = dummy_1_tree.predict(X_train_test)
y_pred_dummy_2 = dummy_2_tree.predict(X_train_test)

# Cálculo del error cuadrático medio.
rmse_tree_dm1 = rmse(y_train_test, y_pred_dummy_1)
rmse_tree_dm2 = rmse(y_train_test, y_pred_dummy_2)  
print(f'\nRMSE (mean): {rmse_tree_dm1}')
print(f'RMSE (median): {rmse_tree_dm2}')

# Cálculo del error absoluto medio.
mae_tree_dm1 = mae(y_train_test, y_pred_dummy_1)
mae_tree_dm2 = mae(y_train_test, y_pred_dummy_2)
print(f'MAE (mean): {mae_tree_dm1}')
print(f'MAE (median): {mae_tree_dm2}')

# Relación entre el error del modelo y el error de los modelos dummy.
print(f'\nRMSE dummy (mean)/RMSE tree: {rmse_tree_dm1/rmse_tree}')
print(f'RMSE dummy (median)/RMSE tree: {rmse_tree_dm2/rmse_tree}')
print(f'MAE dummy (mean)/MAE tree: {mae_tree_dm1/mae_tree}')
print(f'MAE dummy (median)/MAE tree: {mae_tree_dm2/mae_tree}')


# Regresión lineal.
print('\n[bold yellow]Regresión lineal\n------------------[/bold yellow]')

# [Regresión lineal] MODELO BASE.
print('\n[yellow]Modelo base[/yellow]')
base_linear = LinearRegression()

# Entrenamiento (con medición del tiempo).
start = time.time()
base_linear.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_linear = end - start
print(f'Tiempo de entrenamiento: {time_linear:.5f}')

# Predicciones del conjunto de test.
y_pred_n = base_linear.predict(X_train_test_n)

# Denormalizar los datos.
y_pred = scaler.inverse_transform(y_pred_n)

# Cálculo del error cuadrático medio.
rmse_linear = rmse(y_train_test, y_pred)
print(f'\nRMSE: {rmse_linear}')

# Cálculo del error absoluto medio.
mae_linear = mae(y_train_test, y_pred)
print(f'MAE: {mae_linear}')

# [Regresión lineal] MODELO VALIDACIÓN CRUZADA.
print('\n[yellow]Modelo validación cruzada[/yellow]')

'''
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
    print(f'Size of the train set: {train.shape}')
    print(f'Size of the valid set: {valid.shape}')
    print(f'Indices of the train set: {train}')
    print(f'Indices of the valid set: {valid}')
'''

# Usar el predefined split para la validación cruzada.
cv_linear = LinearRegression()

start = time.time()
scores = cross_val_score(cv_linear, X_train_n, y_train_n, cv=ps, scoring='neg_root_mean_squared_error')
end = time.time()
time1_linear_cv = end - start

start = time.time()
scores = cross_val_score(cv_linear, X_train_n, y_train_n, cv=ps, scoring='neg_mean_absolute_error')
end = time.time()
time2_linear_cv = end - start

print(f'Tiempo de entrenamiento (RMSE): {time1_linear_cv:.5f}')
print(f'Tiempo de entrenamiento (MAE): {time2_linear_cv:.5f}')
print(f'\nRMSE: {-scores.mean()}')
print(f'RMSE: {-scores.mean()}')


# [Regresión lineal] MODELOS DUMMY.
print('\n[yellow]Modelos dummy[/yellow]')
dummy_1_linear = DummyRegressor(strategy='mean')
dummy_2_linear = DummyRegressor(strategy='median')

# Entrenamiento de los modelos dummy (con medición del tiempo).
start = time.time()
dummy_1_linear.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_linear_dm1 = end - start

start = time.time()
dummy_2_linear.fit(X_train_train_n, y_train_train_n)
end = time.time()
time_linear_dm2 = end - start

print(f'Tiempo de entrenamiento (mean): {time_linear_dm1:.5f}')
print(f'Tiempo de entrenamiento (median): {time_linear_dm2:.5f}')

# Predicciones del conjunto de test.
y_pred_dummy_1_n = dummy_1_linear.predict(X_train_test_n).reshape(-1, 1)
y_pred_dummy_2_n = dummy_2_linear.predict(X_train_test_n).reshape(-1, 1)

# Denormalizar los datos.
y_pred_dummy_1 = scaler.inverse_transform(y_pred_dummy_1_n)
y_pred_dummy_2 = scaler.inverse_transform(y_pred_dummy_2_n)

# Cálculo del error cuadrático medio.
rmse_linear_dm1 = rmse(y_train_test, y_pred_dummy_1)
rmse_linear_dm2 = rmse(y_train_test, y_pred_dummy_2)
print(f'\nRMSE (mean): {rmse_linear_dm1}')
print(f'RMSE (median): {rmse_linear_dm2}')

# Cálculo del error absoluto medio.
mae_linear_dm1 = mae(y_train_test, y_pred_dummy_1)
mae_linear_dm2 = mae(y_train_test, y_pred_dummy_2)
print(f'MAE (mean): {mae_linear_dm1}')
print(f'MAE (median): {mae_linear_dm2}')

# Cálculo de la diferencia entre el error de los modelos dummy y el modelo base.
print(f'\nRMSE dummy (mean)/RMSE linear: {rmse_linear_dm1/rmse_linear}')
print(f'RMSE dummy (median)/RMSE linear: {rmse_linear_dm2/rmse_linear}')
print(f'MAE dummy (mean)/MAE linear: {mae_linear_dm1/mae_linear}')
print(f'MAE dummy (median)/MAE linear: {mae_linear_dm2/mae_linear}')



#------------------------------------------------------------
'''Evaluación de modelos simples con ajuste de hp.'''
#------------------------------------------------------------

print('\n' + '[bold red]' + '-' * 60 +'\nEvaluación de modelos simples con ajuste de hp.\n' + '-' * 60 + '[/bold red]')

# KNN.
print('\n[bold blue]KNN\n----[/bold blue]')

# Usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split.

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8],
    #'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'leaf_size': [1, 2, 3, 5, 10, 30, 50, 100]
    #'leaf_size': list(range(1,50))
}

# Definimos el modelo.
model = KNeighborsRegressor()

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
grid_result = grid.fit(X_train_n, y_train_n)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_test_n)

# Denormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta].
rmse_knn_a = rmse(y_train_test, y_pred)
print(f'\nRMSE: {rmse_knn_a}')

# Calcular el error absoluto medio en la escala original.
mae_knn_a = mae(y_train_test, y_pred)
print(f'MAE: {mae_knn_a}')

# Mejor score.
#mae_knn_a = -grid_result.best_score_
#print(f'\nMejor score: {-grid_result.best_score_}')


# Árbol de decisión.
print('\n[bold blue]Árbol de decisión\n------------------[/bold blue]')

# Usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split.

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
    #'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Definimos el modelo.
model = DecisionTreeRegressor()
np.random.seed(13)

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
grid_result = grid.fit(X_train, y_train)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_train_test)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta].
rmse_tree_a = rmse(y_train_test, y_pred)
print(f'\nRMSE: {rmse_tree_a}')

# Calcular el error absoluto medio en la escala original.
mae_tree_a = mae(y_train_test, y_pred)
print(f'MAE: {mae_tree_a}')

# Mejor score.
#mae_tree_a = -grid_result.best_score_
#print(f'\nMejor score: {-grid_result.best_score_}')


# Regresión lineal.
print('\n[bold blue]Regresión lineal\n------------------[/bold blue]')

# Usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split.

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'fit_intercept':  [True, False],
    'positive': [True, False],
}

# Definimos el modelo.
model = LinearRegression()

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
grid_result = grid.fit(X_train_n, y_train_n)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_test_n)

# Desnormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta].
rmse_linear_a = rmse(y_train_test, y_pred)
print(f'\nRMSE: {rmse_linear_a}')

# Calcular el error absoluto medio en la escala original.
mae_linear_a = mae(y_train_test, y_pred)
print(f'MAE: {mae_linear_a}')

# Mejor score.
#mae_linear_a = -grid_result.best_score_
#print(f'\nMejor score: {-grid_result.best_score_}')

print()



#------------------------------------------------------------
'''Comparación de modelos simples y resultados.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nComparación de modelos simples y resultados.\n' + '-' * 60 + '[/bold red]')

# KNN.
print('\n[bold green]KNN\n----[/bold green]')

print('MAE sin ajustar:', mae_knn)
print('MAE ajustado:', mae_knn_a)
print('MAE ratio KNN/knn_adjusted:', mae_knn/mae_knn_a)

print('\nRMSE sin ajustar:', rmse_knn)
print('RMSE ajustado:', rmse_knn_a)
print('RMSE ratio KNN/knn_adjusted:', rmse_knn/rmse_knn_a)


# Arbol de decisión.
print('\n[bold green]Árbol de decisión\n------------------[/bold green]')

print('MAE sin ajustar:', mae_tree)
print('MAE ajustado:', mae_tree_a)
print('MAE ratio tree/tree_adjusted:', mae_tree/mae_tree_a)
print('MAE ratio dummy/tree_adjusted:', mae_linear_dm1/mae_tree_a)

print('\nRMSE sin ajustar:', rmse_tree)
print('RMSE ajustado:', rmse_tree_a)
print('RMSE ratio tree/tree_adjusted:', rmse_tree/rmse_tree_a)
print('RMSE ratio dummy/tree_adjusted:', mae_linear_dm2/rmse_tree_a)


# Regresión lineal.
print('\n[bold green]Regresión lineal\n------------------[/bold green]')

print('MAE sin ajustar:', mae_linear)
print('MAE ajustado:', mae_linear_a)
print('MAE ratio linear/linear_adjusted:', mae_linear/mae_linear_a)
#print('(NO VALIDO POR NORMALIZACIÓN) MAE ratio dummy/linear_adjusted:', mae_dummy_linear/mae_linear_adjusted)

print('\nRMSE sin ajustar:', rmse_linear)
print('RMSE ajustado:',rmse_linear_a)
print('RMSE ratio linear/linear_adjusted:', rmse_linear/rmse_linear_a)

print()



#------------------------------------------------------------
'''Reducción de dimensionalidad.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nReducción de dimensionalidad.\n' + '-' * 60 + '[/bold red]')



#------------------------------------------------------------
'''Evaluación de métodos avanzados sin ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de métodos avanzados sin ajuste de hp.\n' + '-' * 60 + '[/bold red]')

print('\n[bold yellow]SVM\n-----[/bold yellow]')

# SVM
svm_model = SVR()
svm_model.fit(X_train_train_n, y_train_train_n.ravel())
svm_preds = svm_model.predict(X_train_test_n)
mae_svm = mae(y_train_test_n, svm_preds)
rmse_svm = rmse(y_train_test_n, svm_preds)

print(f'\nError absoluto medio del modelo SVM: {mae_svm}')
print(f'\nError cuadrático medio del modelo SVM: {rmse_svm}')



print('\n[bold yellow]Random Forests\n---------------[/bold yellow]')

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X_train_train_n, y_train_train_n.ravel())
rf_preds = rf_model.predict(X_train_test_n)
mae_rf = mae(y_train_test_n, rf_preds)
rmse_rf = rmse(y_train_test_n, rf_preds)

print(f'\nError absoluto medio del modelo Random Forests: {mae_rf}')
print(f'\nError cuadrático medio del modelo Random Forests: {rmse_rf}')

#------------------------------------------------------------
'''Evaluación de modelos avanzados con ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de modelos avanzados con ajuste de hp.\n' + '-' * 60 + '[/bold red]')

print('\n[bold blue]SVMs\n-----[/bold blue]')

print('[bold blue]Random Forests\n---------------[/bold blue]')



#------------------------------------------------------------
'''Modelo final.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nModelo final.\n' + '-' * 60)
