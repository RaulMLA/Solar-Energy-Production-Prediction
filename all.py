#!pip install --upgrade linear-tree
#!pip install statsmodels
#!pip install rich

import os

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Métodos avanzados.
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

#PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from pprint import pprint

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

#Crear una carpeta para guardar las gráficas.
'''
if not os.path.exists('graficas'):
    os.makedirs('graficas')

# Boxplot de cada variable meteorológica con seaborn.
sns.set(style='ticks')

# Seleccionar solo las columnas de las variables meteorológicas
cols = mean_df.columns[:-1]

# Crear un diagrama de cajas para cada variable
for col in cols:
    sns.boxplot(data=mean_df, x=col)
    plt.savefig(f'graficas/{col}.png')
    plt.clf()

# Histograma de la variable de respuesta.
sns.displot(mean_df['salida'], kde=True)
plt.savefig('graficas/salida.png')
plt.clf()
'''

# Correlación entre variables meteorológicas.
"""
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
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=0, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .5}, annot=True)

#plt.show()

# Guardar la grafica de correlacion como archivo .jpg.
plt.savefig('correlacion.jpg')

print()


"""
#------------------------------------------------------------
'''División de los datos en entrenamiento y test.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nDivisión de los datos en entrenamiento y test.\n' + '-' * 60 + '[/bold red]')

# Entrenamiento (10 primeros años) y test (2 últimos años).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/12, random_state=13, shuffle=False)

# Comprobamos que los datos se hayan dividido como queremos.
print('Datos de entrenamiento:', X_train.shape, y_train.shape)   # 3650 días -> 10 años.
print('Datos de test:', X_test.shape, y_test.shape)              # 720 días  ->  2 años.

'''
# Convertir dataframe a numpy array.
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
'''

# Normalizamos los datos.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_n = scaler.transform(X_train)
X_test_n = scaler.transform(X_test)

scaler = StandardScaler()
scaler.fit(y_train.values.reshape(-1, 1))
y_train_n = scaler.transform(y_train.values.reshape(-1, 1))
y_test_n = scaler.transform(y_test.values.reshape(-1, 1))

print()



#------------------------------------------------------------
'''Evaluación de modelos simples sin ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de modelos simples sin ajuste de hiperparámetros.\n' + '-' * 60 + '[/bold red]')

# Volvemos a dividir los datos en entrenamiento y test porque la parttición test solo la usaremos en la evaluación final.
X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(X_train, y_train, test_size=1/10, random_state=13, shuffle=False)

# Dividir también los datos normalizados.
X_train_train_n, X_train_validation_n, y_train_train_n, y_train_validation_n = train_test_split(X_train_n, y_train_n, test_size=1/10, random_state=13, shuffle=False)

# Muestra el tamaño de los datos de entrenamiento y test nuevos.
print('Datos train_train: ' , X_train_train.shape, y_train_train.shape)   # 2550 días -> 7 años.
print('Datos train_test: ' , X_train_validation.shape, y_train_validation.shape)      # 1100 días  ->  3 años.


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
print(f'Tiempo de entrenamiento: {time_knn:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred_n = base_knn.predict(X_train_validation_n)

# Denormalizar los datos (aunque se podría RMSE y MAE sin denormalizar).
y_pred = scaler.inverse_transform(y_pred_n)

# Cálculo del error cuadrático medio.
rmse_knn = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_knn}')

# Cálculo del error absoluto medio.
mae_knn = mae(y_train_validation, y_pred)
print(f'MAE: {mae_knn}')


# [KNN] MODELO VALIDACIÓN CRUZADA.
print('\n[yellow]Modelo validación cruzada[/yellow]')

# Usar predefined split para la validación cruzada.

# Número de días de entrenamiento y test.
N_train = 9*365
N_test = 1*365

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

# Denormalizar los datos.
rmse_knn_cv = scaler.inverse_transform(rmse_knn_cv.reshape(-1, 1)).ravel()
mae_knn_cv = scaler.inverse_transform(mae_knn_cv.reshape(-1, 1)).ravel()

print(f'Tiempo de entrenamiento (RMSE): {time1_knn_cv:.5f} segundos.')
print(f'Tiempo de entrenamiento (MAE): {time2_knn_cv:.5f} segundos.')
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

print(f'Tiempo de entrenamiento (mean): {time_knn_dm1:.5f} segundos.')
print(f'Tiempo de entrenamiento (median): {time_knn_dm2:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred_dummy_1_n = dummy_1_knn.predict(X_train_validation_n).reshape(-1, 1)
y_pred_dummy_2_n = dummy_2_knn.predict(X_train_validation_n).reshape(-1, 1)

# Denormalizar los datos (aunque se podría RMSE y MAE sin denormalizar).
y_pred_dummy_1 = scaler.inverse_transform(y_pred_dummy_1_n)
y_pred_dummy_2 = scaler.inverse_transform(y_pred_dummy_2_n)

# Cálculo del error cuadrático medio.
rmse_knn_dm1 = rmse(y_train_validation, y_pred_dummy_1)
rmse_knn_dm2 = rmse(y_train_validation, y_pred_dummy_2)
print(f'\nRMSE (mean): {rmse_knn_dm1}')
print(f'RMSE (median): {rmse_knn_dm2}')

# Cálculo del error absoluto medio.
mae_knn_dm1 = mae(y_train_validation, y_pred_dummy_1)
mae_knn_dm2 = mae(y_train_validation, y_pred_dummy_2)
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
print(f'Tiempo de entrenamiento: {time_tree:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred = base_tree.predict(X_train_validation)

# Cálculo del error cuadrático medio.
rmse_tree = rmse(y_train_validation, y_pred)
print(f'\nError cuadrático medio del modelo Árbol de decisión: {rmse_tree}')

# Cálculo del error absoluto medio.
mae_tree = mae(y_train_validation, y_pred)
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


print(f'Tiempo de entrenamiento: {time1_tree_cv:.5f} segundos.')
print(f'Tiempo de entrenamiento: {time2_tree_cv:.5f} segundos.')
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

print(f'Tiempo de entrenamiento (mean): {time_tree_dm1:.5f} segundos.')
print(f'Tiempo de entrenamiento (median): {time_tree_dm2:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred_dummy_1 = dummy_1_tree.predict(X_train_validation)
y_pred_dummy_2 = dummy_2_tree.predict(X_train_validation)

# Cálculo del error cuadrático medio.
rmse_tree_dm1 = rmse(y_train_validation, y_pred_dummy_1)
rmse_tree_dm2 = rmse(y_train_validation, y_pred_dummy_2)  
print(f'\nRMSE (mean): {rmse_tree_dm1}')
print(f'RMSE (median): {rmse_tree_dm2}')

# Cálculo del error absoluto medio.
mae_tree_dm1 = mae(y_train_validation, y_pred_dummy_1)
mae_tree_dm2 = mae(y_train_validation, y_pred_dummy_2)
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
print(f'Tiempo de entrenamiento: {time_linear:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred_n = base_linear.predict(X_train_validation_n)

# Denormalizar los datos.
y_pred = scaler.inverse_transform(y_pred_n)

# Cálculo del error cuadrático medio.
rmse_linear = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_linear}')

# Cálculo del error absoluto medio.
mae_linear = mae(y_train_validation, y_pred)
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
rmse_linear_cv = cross_val_score(cv_linear, X_train_n, y_train_n, cv=ps, scoring='neg_root_mean_squared_error')
end = time.time()
time1_linear_cv = end - start

start = time.time()
mae_linear_cv = cross_val_score(cv_linear, X_train_n, y_train_n, cv=ps, scoring='neg_mean_absolute_error')
end = time.time()
time2_linear_cv = end - start

# Denormalizar los datos.
rmse_linear_cv = scaler.inverse_transform(rmse_linear_cv.reshape(-1, 1)).ravel()
mae_linear_cv = scaler.inverse_transform(mae_linear_cv.reshape(-1, 1)).ravel()

print(f'Tiempo de entrenamiento (RMSE): {time1_linear_cv:.5f} segundos.')
print(f'Tiempo de entrenamiento (MAE): {time2_linear_cv:.5f} segundos.')
print(f'\nRMSE: {-rmse_linear_cv.mean()}')
print(f'RMSE: {-mae_linear_cv.mean()}')


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

print(f'Tiempo de entrenamiento (mean): {time_linear_dm1:.5f} segundos.')
print(f'Tiempo de entrenamiento (median): {time_linear_dm2:.5f} segundos.')

# Predicciones del conjunto de test.
y_pred_dummy_1_n = dummy_1_linear.predict(X_train_validation_n).reshape(-1, 1)
y_pred_dummy_2_n = dummy_2_linear.predict(X_train_validation_n).reshape(-1, 1)

# Denormalizar los datos.
y_pred_dummy_1 = scaler.inverse_transform(y_pred_dummy_1_n)
y_pred_dummy_2 = scaler.inverse_transform(y_pred_dummy_2_n)

# Cálculo del error cuadrático medio.
rmse_linear_dm1 = rmse(y_train_validation, y_pred_dummy_1)
rmse_linear_dm2 = rmse(y_train_validation, y_pred_dummy_2)
print(f'\nRMSE (mean): {rmse_linear_dm1}')
print(f'RMSE (median): {rmse_linear_dm2}')

# Cálculo del error absoluto medio.
mae_linear_dm1 = mae(y_train_validation, y_pred_dummy_1)
mae_linear_dm2 = mae(y_train_validation, y_pred_dummy_2)
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

'''
La principal diferencia entre Grid Search y Randomized Search es la forma en que exploran el espacio de
hiperparámetros. Grid Search explora todas las combinaciones posibles de los valores especificados para
cada hiperparámetro. Es decir, crea una malla (o grid) de valores posibles para cada hiperparámetro y
prueba todas las combinaciones. Es una estrategia exhaustiva que garantiza que se probarán todas las
combinaciones posibles, pero puede ser extremadamente costosa computacionalmente, especialmente cuando
el número de hiperparámetros es grande.

Por otro lado, Randomized Search es un enfoque más eficiente que selecciona aleatoriamente combinaciones
de hiperparámetros para probar en lugar de probar todas las combinaciones posibles. Esto hace que Randomized
Search sea más rápido y escalable para espacios de hiperparámetros grandes. Sin embargo, no garantiza que
se prueben todas las combinaciones posibles, lo que puede resultar en que se omitan combinaciones importantes.

En general, se recomienda utilizar Randomized Search cuando el espacio de hiperparámetros es grande y
se dispone de recursos limitados para la búsqueda, mientras que Grid Search es preferible cuando el
espacio de hiperparámetros es pequeño y se dispone de suficientes recursos computacionales para realizar
una búsqueda exhaustiva.
'''

print('\n' + '[bold red]' + '-' * 60 +'\nEvaluación de modelos simples con ajuste de hp.\n' + '-' * 60 + '[/bold red]')

# KNN.
print('\n[bold blue]KNN\n----[/bold blue]')

# Usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split.

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6],
    #'n_neighbors': [1, 2, 5, 8, 10, 15, 20, 40],
    #'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'leaf_size': [1, 2, 5, 10, 20],
    #'leaf_size': list(range(1,50)),
    'p':[1, 2]
}

# Definimos el modelo.
model = KNeighborsRegressor()

# Definir la estrategia de validación cruzada
#cv = TimeSeriesSplit(n_splits=3)

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
start = time.time()
grid_result = grid.fit(X_train_n, y_train_n)
end = time.time()
t_knn_gs = end - start
print(f'Tiempo de entrenamiento (Grid Search): {t_knn_gs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_validation_n)

# Denormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
rmse_knn_a = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_knn_a}')

# Calcular el error absoluto medio en la escala original.
mae_knn_a = mae(y_train_validation, y_pred)
print(f'MAE: {mae_knn_a}')


# Ahora, usaremos random search para comparar resultados con grid search.
r_search = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              cv=ps,
                              scoring='neg_mean_absolute_error',
                              verbose=1,
                              n_jobs=-1,
                              n_iter=10)


# Entrenamos el random search.
start = time.time()
r_search_result = r_search.fit(X_train_n, y_train_n)
end = time.time()
t_knn_rs = end - start
print(f'\nTiempo de entrenamiento (Randomized Search): {t_knn_rs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {r_search_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = r_search_result.best_estimator_
y_pred_n = best_model.predict(X_train_validation_n)

# Denormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
rmse_knn_b = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_knn_b}')

# Calcular el error absoluto medio en la escala original.
mae_knn_b = mae(y_train_validation, y_pred)
print(f'MAE: {mae_knn_b}')


# Mejor score.
#mae_knn_a = -grid_result.best_score_
#print(f'\nMejor score: {-grid_result.best_score_}')


# Árbol de decisión.
print('\n[bold blue]Árbol de decisión\n------------------[/bold blue]')

# Usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split.

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'max_depth': [1, 2, 5, 8, 10, 15, 20, 40],
     #'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 5, 8, 10, 15, 20, 40],
}

# Definimos el modelo.
model = DecisionTreeRegressor()

# Definir la estrategia de validación cruzada
#cv = TimeSeriesSplit(n_splits=3)

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
start = time.time()
grid_result = grid.fit(X_train, y_train)
end = time.time()
t_tree_gs = end - start
print(f'Tiempo de entrenamiento (Grid Search): {t_tree_gs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_train_validation)

# Calcular el error cuadrático medio en la escala original.
rmse_tree_a = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_tree_a}')

# Calcular el error absoluto medio en la escala original.
mae_tree_a = mae(y_train_validation, y_pred)
print(f'MAE: {mae_tree_a}')


# Ahora, usaremos random search para comparar resultados con grid search.
r_search = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              cv=ps,
                              scoring='neg_mean_absolute_error',
                              verbose=1,
                              n_jobs=-1,
                              n_iter=10)

# Entrenamos el grid search.
start = time.time()
r_search_result = r_search.fit(X_train, y_train)
end = time.time()
t_tree_rs = end - start
print(f'\nTiempo de entrenamiento (Randomized Search): {t_tree_rs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {r_search_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = r_search_result.best_estimator_
y_pred = best_model.predict(X_train_validation)

# Calcular el error cuadrático medio en la escala original.
rmse_tree_b = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_tree_b}')

# Calcular el error absoluto medio en la escala original.
mae_tree_b = mae(y_train_validation, y_pred)
print(f'MAE: {mae_tree_b}')


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

# Definir la estrategia de validación cruzada
#cv = TimeSeriesSplit(n_splits=3)

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
start = time.time()
grid_result = grid.fit(X_train_n, y_train_n)
end = time.time()
t_linear_gs = end - start
print(f'Tiempo de entrenamiento (Grid Search): {t_linear_gs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_validation_n)

# Desnormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
rmse_linear_a = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_linear_a}')

# Calcular el error absoluto medio en la escala original.
mae_linear_a = mae(y_train_validation, y_pred)
print(f'MAE: {mae_linear_a}')

# Ahora, usaremos random search para comparar resultados con grid search.
r_search = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              cv=ps,
                              scoring='neg_mean_absolute_error',
                              verbose=1,
                              n_jobs=-1,
                              n_iter=4)

# Entrenamos el grid search.
start = time.time()
r_search_result = r_search.fit(X_train_n, y_train_n)
end = time.time()
t_linear_rs = end - start
print(f'\nTiempo de entrenamiento (Randomized Search): {t_linear_rs:.5f} segundos.')

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {r_search_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = r_search_result.best_estimator_
y_pred_n = best_model.predict(X_train_validation_n)

# Desnormalizar la predicción del modelo.
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
rmse_linear_b = rmse(y_train_validation, y_pred)
print(f'\nRMSE: {rmse_linear_b}')

# Calcular el error absoluto medio en la escala original.
mae_linear_b = mae(y_train_validation, y_pred)
print(f'MAE: {mae_linear_b}')


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
print(f'\nMAE dummy (mean)/MAE KNN: {mae_knn_dm1/mae_knn_a}')
print(f'MAE dummy (median)/MAE KNN: {mae_knn_dm2/mae_knn_a}')

print('\nRMSE sin ajustar:', rmse_knn)
print('RMSE ajustado:', rmse_knn_a)
print('RMSE ratio KNN/knn_adjusted:', rmse_knn/rmse_knn_a)
print(f'\nRMSE dummy (mean)/RMSE KNN: {rmse_knn_dm1/rmse_knn_a}')
print(f'RMSE dummy (median)/RMSE KNN: {rmse_knn_dm2/rmse_knn_a}')


# Arbol de decisión.
print('\n[bold green]Árbol de decisión\n------------------[/bold green]')

print('MAE sin ajustar:', mae_tree)
print('MAE ajustado:', mae_tree_a)
print('MAE ratio tree/tree_adjusted:', mae_tree/mae_tree_a)
print(f'\nMAE dummy (mean)/MAE tree: {mae_tree_dm1/mae_tree_a}')
print(f'MAE dummy (median)/MAE tree: {mae_tree_dm2/mae_tree_a}')

print('\nRMSE sin ajustar:', rmse_tree)
print('RMSE ajustado:', rmse_tree_a)
print('RMSE ratio tree/tree_adjusted:', rmse_tree/rmse_tree_a)
print(f'\nRMSE dummy (mean)/RMSE tree: {rmse_tree_dm1/rmse_tree_a}')
print(f'RMSE dummy (median)/RMSE tree: {rmse_tree_dm2/rmse_tree_a}')


# Regresión lineal.
print('\n[bold green]Regresión lineal\n------------------[/bold green]')

print('MAE sin ajustar:', mae_linear)
print('MAE ajustado:', mae_linear_a)
print('MAE ratio linear/linear_adjusted:', mae_linear/mae_linear_a)
print(f'\nMAE dummy (mean)/MAE linear: {mae_linear_dm1/mae_linear_a}')
print(f'MAE dummy (median)/MAE linear: {mae_linear_dm2/mae_linear_a}')

print('\nRMSE sin ajustar:', rmse_linear)
print('RMSE ajustado:',rmse_linear_a)
print('RMSE ratio linear/linear_adjusted:', rmse_linear/rmse_linear_a)
print(f'\nRMSE dummy (mean)/RMSE linear: {rmse_linear_dm1/rmse_linear_a}')
print(f'RMSE dummy (median)/RMSE linear: {rmse_linear_dm2/rmse_linear_a}')

print()



#------------------------------------------------------------
'''Reducción de dimensionalidad.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nReducción de dimensionalidad.\n' + '-' * 60 + '[/bold red]')

# Quitamos del dataframe de las medias de las variables que no queremos usar.
#df_reducida = mean_df.drop(['apcp_sf_media', 'pres_msl_media', 'tcdc_eatm_media', 'tcolc_eatm_media'], axis=1)
#df_reducida = mean_df.drop(['tmin_2m_media', 'tmp_2m_media', 'tmp_sfc_media', 'tmax_2m_media', 'tcolc_eatm_media', 'spfh_2m_media', 'pwat_eatm_media'], axis=1)
df_reducida=mean_df

#df_reducida=disp_df.drop(['tmin_2m1_1', 'tmin_2m2_1', 'tmin_2m3_1', 'tmin_2m4_1', 'tmin_2m5_1', 'tmp_2m_1_1', 'tmp_2m_2_1', 'tmp_2m_3_1', 'tmp_2m_4_1', 'tmp_2m_5_1', 'tmp_sfc1_1', 'tmp_sfc2_1', 'tmp_sfc3_1', 'tmp_sfc4_1', 'tmp_sfc5_1', 'tmax_2m1_1', 'tmax_2m2_1', 'tmax_2m3_1', 'tmax_2m4_1', 'tmax_2m5_1', 'tcolc_e1_1', 'tcolc_e2_1', 'tcolc_e3_1', 'tcolc_e4_1', 'tcolc_e5_1', 'spfh_2m1_1', 'spfh_2m2_1', 'spfh_2m3_1', 'spfh_2m4_1', 'spfh_2m5_1', 'pwat_ea1_1', 'pwat_ea2_1', 'pwat_ea3_1', 'pwat_ea4_1', 'pwat_ea5_1'], axis=1)
# Imprimimos el dataframe reducido.
print(df_reducida)

print(df_reducida.shape)

# Dividimos el dataframe reducido en train y test.
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(df_reducida.drop('salida', axis=1), df_reducida['salida'], test_size=2/12, random_state=13, shuffle=False)

# Volvemos a dividir el train en train_train y train_test.
X_train_train_r, X_train_validation_r, y_train_train_r, y_train_validation_r = train_test_split(X_train_r, y_train_r, test_size=1/10, random_state=13, shuffle=False)

# Normalizamos los datos.
scaler_r = MinMaxScaler()
scaler_r.fit(X_train_train_r)
X_train_r_n = scaler_r.transform(X_train_r)
X_train_train_r_n = scaler_r.transform(X_train_train_r)
X_train_validation_r_n = scaler_r.transform(X_train_validation_r)
X_test_r_n = scaler_r.transform(X_test_r)

# Normalizamos la salida.
scaler_r = MinMaxScaler()
scaler_r.fit(y_train_train_r.values.reshape(-1, 1))
y_train_r_n = scaler_r.transform(y_train_r.values.reshape(-1, 1))
y_train_train_r_n = scaler_r.transform(y_train_train_r.values.reshape(-1, 1))
y_train_validation_r_n = scaler_r.transform(y_train_validation_r.values.reshape(-1, 1))
y_test_r_n = scaler_r.transform(y_test_r.values.reshape(-1, 1))

# Entrenamos a los modelos Knn, Árbol de decisión y Regresión lineal con el dataframe reducido y con los mejores hiperparametros.

# KNN.
print('\n[bold green]KNN\n-----[/bold green]')
knn_model_r = KNeighborsRegressor()
start = time.time()
knn_model_r.fit(X_train_train_r_n, y_train_train_r_n)
end = time.time()
time_knn_r = end - start
print(f'Tiempo de entrenamiento: {time_knn_r:.5f} segundos.')
knn_preds_r_n = knn_model_r.predict(X_train_validation_r_n)
knn_preds_r = scaler_r.inverse_transform(knn_preds_r_n)
mae_knn_r = mae(y_train_validation_r, knn_preds_r)
rmse_knn_r = rmse(y_train_validation_r, knn_preds_r)

print(f'\nRMSE: {rmse_knn_r}')
print(f'MAE: {mae_knn_r}')

#ajuste de hiperparametros
print('\n[bold green]Ajuste de hiperparámetros KNN\n---------------------------[/bold green]')

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    #'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8],
    'n_neighbors': [1, 2, 5, 8, 10, 15, 20, 40],
    #'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'leaf_size': [1, 2, 5, 10, 100]
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
grid_result = grid.fit(X_train_r_n, y_train_r_n)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados.
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_validation_r_n)

# Denormalizar la predicción del modelo.
y_pred = scaler_r.inverse_transform(y_pred_n)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta.
rmse_knn_a_r = rmse(y_train_validation_r, y_pred)
print(f'\nRMSE ajustado: {rmse_knn_a_r}')

# Calcular el error absoluto medio en la escala original.
mae_knn_a_r = mae(y_train_validation_r, y_pred)
print(f'MAE ajustado: {mae_knn_a_r}')

# Arbol de decisión.
print('\n[bold green]Árbol de decisión\n------------------[/bold green]')
tree_model_r = DecisionTreeRegressor()
start = time.time()
tree_model_r.fit(X_train_train_r, y_train_train_r)
end = time.time()
time_tree_r = end - start
print(f'Tiempo de entrenamiento: {time_tree_r:.5f} segundos.')
tree_preds_r = tree_model_r.predict(X_train_validation_r)
mae_tree_r = mae(y_train_validation_r, tree_preds_r)
rmse_tree_r = rmse(y_train_validation_r, tree_preds_r)

print(f'\nRMSE: {rmse_tree_r}')
print(f'MAE: {mae_tree_r}')

#ajuste de hiperparametros
print('\n[bold green]Ajuste de hiperparámetros Árbol de decisión\n---------------------------[/bold green]')

# Definimos el diccionario de los valores de los hiperparámetros que queremos probar.
param_grid = {
    'max_depth': [1, 2, 5, 8, 10, 15, 20, 40],
     #'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [1, 2, 5, 8, 10, 15, 20, 40],
}

# Definimos el modelo.
model = DecisionTreeRegressor()

# Definimos el grid search.
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=ps,
                    scoring='neg_mean_absolute_error',
                    verbose=1,
                    n_jobs=-1)

# Entrenamos el grid search.
grid_result = grid.fit(X_train_r, y_train_r)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba.
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_train_validation_r)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta.
rmse_tree_a_r = rmse(y_train_validation_r, y_pred)
print(f'\nRMSE ajustado: {rmse_tree_a_r}')

# Calcular el error absoluto medio en la escala original.
mae_tree_a_r = mae(y_train_validation_r, y_pred)
print(f'MAE ajustado: {mae_tree_a_r}')

# Regresión lineal.
print('\n[bold green]Regresión lineal\n------------------[/bold green]')
linear_model_r = LinearRegression()
start = time.time()
linear_model_r.fit(X_train_train_r_n, y_train_train_r_n)
end = time.time()
time_linear_r = end - start
print(f'Tiempo de entrenamiento: {time_linear_r:.5f} segundos.')
linear_preds_r_n = linear_model_r.predict(X_train_validation_r_n)
linear_preds_r = scaler_r.inverse_transform(linear_preds_r_n)
mae_linear_r = mae(y_train_validation_r, linear_preds_r)
rmse_linear_r = rmse(y_train_validation_r, linear_preds_r)

print(f'\nRMSE: {rmse_linear_r}')
print(f'MAE: {mae_linear_r}')

#ajuste de hiperparametros
print('\n[bold green]Ajuste de hiperparámetros Regresión lineal\n---------------------------[/bold green]')
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
grid_result = grid.fit(X_train_r_n, y_train_r_n)

# Mejores hiperparámetros.
print(f'\nMejores hiperparámetros: {grid_result.best_params_}')

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba.
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_train_validation_r_n)

# Calcular el error cuadrático medio en la escala original.
# Con scoring="neg_mean_absolute_error" en GridSearch creo que no hace falta.
rmse_linear_a_r = rmse(y_train_validation_r, y_pred)
print(f'\nRMSE ajustado: {rmse_linear_a_r}')

# Calcular el error absoluto medio en la escala original.
mae_linear_a_r = mae(y_train_validation_r, y_pred)
print(f'MAE ajustado: {mae_linear_a_r}')


"""
#------------------------------------------------------------
'''Evaluación de métodos avanzados sin ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de métodos avanzados sin ajuste de hp.\n' + '-' * 60 + '[/bold red]')

# SVM
print('\n[bold yellow]SVM\n-----[/bold yellow]')

svm_model = SVR()
start = time.time()
svm_model.fit(X_train_train_n, y_train_train_n.ravel())
end = time.time()
time_svm = end - start
print(f'Tiempo de entrenamiento: {time_svm:.5f} segundos.')
svm_preds = svm_model.predict(X_train_validation_n)
svm_preds = scaler.inverse_transform(svm_preds.reshape(-1,1))
mae_svm = mae(y_train_validation, svm_preds)
rmse_svm = rmse(y_train_validation, svm_preds)

print(f'\nMAE: {mae_svm}')
print(f'\nRMSE: {rmse_svm}')


# Random Forest
print('\n[bold yellow]Random Forests\n---------------[/bold yellow]')

rf_model = RandomForestRegressor()
start = time.time()
rf_model.fit(X_train_train_n, y_train_train_n.ravel())
end = time.time()
time_forest= end - start
print(f'Tiempo de entrenamiento: {time_forest:.5f} segundos.')
rf_preds = rf_model.predict(X_train_validation_n)
rf_preds = scaler.inverse_transform(rf_preds.reshape(-1,1))
mae_rf = mae(y_train_validation, rf_preds)
rmse_rf = rmse(y_train_validation, rf_preds)

print(f'\nMAE: {mae_rf}')
print(f'\nRMSE: {rmse_rf}')

#------------------------------------------------------------
'''Evaluación de modelos avanzados con ajuste de hp.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nEvaluación de modelos avanzados con ajuste de hp.\n' + '-' * 60 + '[/bold red]')

print('\n[bold blue]SVMs\n-----[/bold blue]')

#ajuste de hiperparametros
svm_model = SVR()
svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 0.5, 1, 2],
                'gamma': ['scale', 'auto'],
                'degree': [1, 2, 3],
                'coef0': [0.1, 0.5, 1],
                'epsilon': [0.1, 0.5, 1, 5],
                #'shrinking': [True, False],
                'tol': [0.001, 0.0001, 0.00001],
                #'cache_size': [200, 500, 1000],
                #'max_iter': [-1, 1000, 2000, 5000, 10000]
                }

svm_grid = GridSearchCV(svm_model, svm_params, cv=ps, n_jobs=-1, verbose=1)
start = time.time()
svm_grid.fit(X_train_n, y_train_n.ravel())
end = time.time()
time_svm_a = end - start
print(f'Tiempo de entrenamiento: {time_svm_a:.5f} segundos.')
print("Mejores hiperparámetros:",svm_grid.best_params_)
svm_preds = svm_grid.predict(X_train_validation_n)
svm_preds = scaler.inverse_transform(svm_preds.reshape(-1,1))
mae_svm_a = mae(y_train_validation, svm_preds)
rmse_svm_a = rmse(y_train_validation, svm_preds)

print(f'\nMAE: {mae_svm_a}')
print(f'\nRMSE: {rmse_svm_a}')


print('\n[bold blue]Random Forests\n---------------[/bold blue]')

#ajuste de hiperparametros
rf_model = RandomForestRegressor()
rf_params = {   'n_estimators': [100, 200, 400, 500],
                #'n_estimators': [100, 200, 500],
                #'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'max_depth': [None, 1, 2, 5],
                #'max_depth': [None, 2, 5, 10],
                #'min_samples_split': [1, 2, 3],
                #'min_samples_leaf': [1, 5, 10],
                #'max_features': ['auto', 'sqrt', 'log2'],
                #'max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                #'min_impurity_decrease': [0.0, 0.2, 0.4, 0.5, 0.9],
                'bootstrap': [True, False],
                #'oob_score': [True, False],
                #'warm_start': [True, False],
                #'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                #'max_samples': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                }

rf_grid = GridSearchCV(rf_model, rf_params, cv=ps, n_jobs=-1, verbose=1)
start = time.time()
rf_grid.fit(X_train, y_train.ravel())
end = time.time()
time_forest_a = end - start
print(f'Tiempo de entrenamiento: {time_forest_a:.5f} segundos.')
print("Mejores hiperparámetros:",rf_grid.best_params_)
rf_preds = rf_grid.predict(X_train_validation)
#rf_preds = scaler.inverse_transform(rf_preds.reshape(-1,1))
mae_rf_a = mae(y_train_validation, rf_preds)
rmse_rf_a = rmse(y_train_validation, rf_preds)

print(f'\nMAE: {mae_rf_a}')
print(f'\nRMSE: {rmse_rf_a}')

#------------------------------------------------------------
'''Importancia de variables.'''
#------------------------------------------------------------


print('[bold red]' + '-' * 60 +'\nImportancia de variables.\n' + '-' * 60 + '[/bold red]')

print('\n[bold blue]Random Forests\n---------------[/bold blue]')
print(f'\nImportancia de variables del modelo Random Forests: {rf_grid.best_estimator_.feature_importances_}')

#PCA 
random_forest=RandomForestRegressor(n_estimators=500, max_depth=None, bootstrap=True)
selector = SelectKBest(f_regression)

pipeline = Pipeline([('select', selector), ('random_forest', random_forest)])

param_grid = {'select__k': list(range(1,15))}
tune_select_rf = GridSearchCV(pipeline,
                                     param_grid,
                                     scoring="neg_mean_absolute_error",
                                     cv=ps
                                     )

tune_select_rf.fit(X_train, y_train.ravel())

print(tune_select_rf.best_params_, np.sqrt(-tune_select_rf.best_score_))

trained_pipeline = tune_select_rf.best_estimator_

print(f"Features selected: {trained_pipeline.named_steps['select'].get_support()}")

print(f"Locations where features selected: {np.where(trained_pipeline.named_steps['select'].get_support())}")


feature_names_before_selection = disp_df.drop('salida', axis=1).columns

print(f"In Scikit-learn 1.x, we can even get the feature names after selection: {trained_pipeline.named_steps['select'].get_feature_names_out(feature_names_before_selection)}")

pprint(list(zip(tune_select_rf.cv_results_['param_select__k'].data, -tune_select_rf.cv_results_['mean_test_score'])))

plt.plot(tune_select_rf.cv_results_['param_select__k'].data, -tune_select_rf.cv_results_['mean_test_score'])
plt.ylabel('SCORE')
plt.xlabel('Number of features')
#guardar imagen
plt.savefig('pca.png')

predictions_test = tune_select_rf.predict(X_train_validation)
mae_rf_a_r = mae(y_train_validation, predictions_test)

print(f'\nMAE: {mae_rf_a_r}')

#------------------------------------------------------------
'''Comparación de modelos avanzado y resultados.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nComparación de modelos avanzado y resultados.\n' + '-' * 60 + '[/bold red]')
print('\n[bold blue]SVMs\n-----[/bold blue]')
print(f'\nError absoluto medio del modelo SVM: {mae_svm}')
print(f'\nError cuadrático medio del modelo SVM: {rmse_svm}')

print(f'\nRatio de mejora del error absoluto medio del modelo SVM: {mae_svm/mae_svm_a}')

print('\n[bold blue]Random Forests\n---------------[/bold blue]')
print(f'\nError absoluto medio del modelo Random Forests: {mae_rf}')
print(f'\nError cuadrático medio del modelo Random Forests: {rmse_rf}')

print(f'\nRatio de mejora del error absoluto medio del modelo Random Forests: {mae_rf/mae_rf_a}')

#------------------------------------------------------------
'''Modelo final.'''
#------------------------------------------------------------

print('[bold red]' + '-' * 60 +'\nModelo final.\n' + '-' * 60)
"""