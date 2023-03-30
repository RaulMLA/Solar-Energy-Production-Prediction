#!pip install --upgrade linear-tree
#!pip install statsmodels

#importación de datos
import numpy as np
import pandas as pd

#EDA
import seaborn as sns
import matplotlib.pyplot as plt

#División de los datos en entrenamiento y test
from sklearn.model_selection import train_test_split

#Evaluación de modelos sin ajuste de hiperparámetros
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import time

from sklearn.model_selection import PredefinedSplit, cross_val_score

from sklearn.dummy import DummyRegressor

#Evaluación de modelos con ajuste de hiperparámetros
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------
"""Importamos los datos."""
#------------------------------------------------------------

print("-" * 60)
print("Importando los datos...")
print("-" * 60)

# Datos disponibles.
disp_df = pd.read_csv("disp_st13ns1.txt.bz2",
                      compression="bz2",
                      index_col=0)

# Datos competición.
comp_df = pd.read_csv("comp_st13ns1.txt.bz2",
                      compression="bz2",
                      index_col=0)

# Mostramos la información de cada conjunto de datos.
print(f"El conjunto de datos disponibles tiene {len(disp_df)} instancias.")
print(f"El conjunto de datos de competición tiene {len(disp_df)} instancias.")


# Datos.
X = disp_df.drop('salida', axis=1)

# Etiquetas.
y = disp_df.salida

print()



#------------------------------------------------------------
"""EDA"""
#------------------------------------------------------------

print("-" * 60)
print("Análisis Exploratorio de Datos (EDA)")
print("-" * 60)

# Mostramos todos los datos.
print('Matriz de atributos:\n\n', X)
print('\n\nVector de la variable de respuesta:\n\n', y)

# Mostramos el tipo de dato de una variable meteorológica y de un valor de la variable de respuesta.
print('\nEjemplo de tipo de dato de variable meteorológica:', type(X['apcp_sf1_1'][0]))
print('Ejemplo de tipo de dato de variable de respuesta:', type(y[0]))

# Contamos el número de missing values.
print("\nMissing values: ", disp_df.isnull().values.sum())

#Media de cada variable meteorológica
variables_meteorologicas = ['apcp_sf', 'dlwrf_s', 'dswrf_s', 'pres_msl', 'pwat_eatm', 'spfh_2m', 'tcdc_eatm', 'tcolc_eatm', 'tmax_2m', 'tmin_2m', 'tmp_2m', 'tmp_sfc', 'ulwrf_sfc', 'ulwrf_tatm', 'uswrf_sfc']

#Creamos un dataframe con la media de cada variable meteorológica
mean_df = disp_df.iloc[:, :-1].groupby(np.arange(len(disp_df.columns)-1)//5, axis=1).mean()
mean_df.columns = [f'{name}_media' for name in variables_meteorologicas]
mean_df['salida'] = disp_df['salida']
#print(mean_df)

#Boxplot de cada variable meteorológica con seaborn
"""
sns.set(style="ticks")

# Seleccionar solo las columnas de las variables meteorológicas
cols = mean_df.columns[:-1]

# Crear un diagrama de cajas para cada variable
for col in cols:
    sns.histplot(data=mean_df, x=col)
    plt.show()
"""

#Correlación entre variables meteorológicas

sns.set_theme(style="white")

# Computar la matriz de correlación
corr = mean_df.corr()

#print(corr)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

#plt.show()

#sacar la grafica de correlacion en un archivo jpg
plt.savefig('correlacion.jpg')

print()



#------------------------------------------------------------
"""División de los datos en entrenamiento y test."""
#------------------------------------------------------------

print("-" * 60)
print("División de los datos en entrenamiento y test.")
print("-" * 60)

# Entrenamiento (10 primeros años) y test (2 últimos años).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/12, random_state=13, shuffle=False)

# Comprobamos que los datos se hayan dividido como queremos.
print("Datos de entrenamiento:", X_train.shape, y_train.shape)   # 3650 días -> 10 años.
print("Datos de test:", X_test.shape, y_test.shape)     # 720 días  ->  2 años.

#Normalizamos los datos
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
"""Evaluación de modelos sin ajuste de hiperparámetros."""
#------------------------------------------------------------

print("-" * 60)
print("Evaluación de modelos sin ajuste de hiperparámetros.")
print("-" * 60)

#Volvemos a dividir los datos en entrenamiento y test porque la parttición test solo la usaremos en la evaluación final
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=3/10, random_state=13, shuffle=False)

#Dividir también los datos normalizados
X_train_train_n, X_train_test_n, y_train_train_n, y_train_test_n = train_test_split(X_train_n, y_train_n, test_size=3/10, random_state=13, shuffle=False)

#muestra el tamaño de los datos de entrenamiento y test nuevos
print("Datos train_train: " ,X_train_train.shape, y_train_train.shape)   # 2550 días -> 7 años.
print("Datos train_test: " ,X_train_test.shape, y_train_test.shape)     # 1100 días  ->  3 años.

#KNN
print("\nKNN\n----")

# Entrenamiento del modelo.
clf = KNeighborsRegressor()
np.random.seed(13)

#medir tiempo de entrenamiento
start = time.time()
clf.fit(X_train_train_n, y_train_train_n)
end = time.time()
print(f"Tiempo de entrenamiento: {(end - start):.5f}")

y_pred = clf.predict(X_train_test_n)

#Desnormalizar los datos
y_pred = scaler.inverse_transform(y_pred)

#Calculo del error cuadrático medio
rmse_knn = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"\nError cuadrático medio del modelo KNN: {rmse_knn}")

#Calculo del error absoluto medio
mae_knn = metrics.mean_absolute_error(y_train_test, y_pred)
print(f"Error absoluto medio del modelo KNN: {mae_knn}")
np.random.seed(13)

#Usar predefined split para la validación cruzada

print("\nValidación cruzada con PredefinedSplit\n")
N_train = 7*365
N_test = 3*365

selector = [-1] * N_train + [0] * N_test
ps = PredefinedSplit(selector)

"""
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
  print(f"Size of the train set: {train.shape}")
  print(f"Size of the valid set: {valid.shape}")
  print(f"Indices of the train set: {train}")
  print(f"Indices of the valid set: {valid}")
"""

#usar el predefined split para la validación cruzada
clf = KNeighborsRegressor()

np.random.seed(13)
scores = cross_val_score(clf, X_train_n, y_train_n, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo KNN calculado con validación cruzada: {-scores.mean()}")

np.random.seed(13)
scores = cross_val_score(clf, X_train_n, y_train_n, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo KNN calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Árbol de decisión
print("\nÁrbol de decisión\n------------------")

# Entrenamiento del modelo.
clf = tree.DecisionTreeRegressor()
np.random.seed(13)

#medir tiempo de entrenamiento
start = time.time()
clf.fit(X_train_train, y_train_train)
end = time.time()
print(f"Tiempo de entrenamiento: {(end - start):.5f}")

# Predicciones del conjunto de test.
y_pred = clf.predict(X_train_test)

#Calculo del error cuadrático medio
rmse_tree = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"\nError cuadrático medio del modelo Árbol de decisión: {rmse_tree}")

#Calculo del error absoluto medio
mae_tree = metrics.mean_absolute_error(y_train_test, y_pred)
print(f"Error absoluto medio del modelo Árbol de decisión: {mae_tree}")
np.random.seed(13)

#Usar predefined split para la validación cruzada
print("\nValidación cruzada con PredefinedSplit\n")

ps = PredefinedSplit(selector)

"""
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
    print(f"Size of the train set: {train.shape}")
    print(f"Size of the valid set: {valid.shape}")
    print(f"Indices of the train set: {train}")
    print(f"Indices of the valid set: {valid}")
"""

#usar el predefined split para la validación cruzada
clf = tree.DecisionTreeRegressor()

np.random.seed(13)
scores = cross_val_score(clf, X_train, y_train, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo Árbol de decisión calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

scores = cross_val_score(clf, X_train, y_train, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo Árbol de decisión calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Probamos si es mejor que un dummy regressor tree

#Estrategy mean
regr_mean = DummyRegressor(strategy="mean")
regr_mean.fit(X_train_train, y_train_train)
rmse_mean = np.sqrt(metrics.mean_squared_error(y_train_test, regr_mean.predict(X_train_test)))

print("\nDummy Regressor:\n")
print(f"Error cuadrático medio del arbol dummy (mean): {rmse_mean}")
print(f"RMSE ratio tree/dummy(mean): {rmse_mean/rmse_tree}")

#Estrategy median
regr_median = DummyRegressor(strategy="median")
regr_median.fit(X_train_train, y_train_train)
mae_dummy_tree = metrics.mean_absolute_error(y_train_test, regr_median.predict(X_train_test))

print(f"\nError absoluto medio del arbol dummy (median): {mae_dummy_tree}")
print(f"MAE ratio tree/dummy(median): {mae_dummy_tree/mae_tree}")

#Regresión lineal
print("\nRegresión lineal\n------------------")

# Entrenamiento del modelo.
clf = LinearRegression()
np.random.seed(13)

#medir tiempo de entrenamiento
start = time.time()
clf.fit(X_train_train_n, y_train_train_n)
end = time.time()
print(f"Tiempo de entrenamiento: {(end - start):.5f}")

# Predicciones del conjunto de test.
y_pred = clf.predict(X_train_test_n)

#Desnormalizar los datos
y_pred = scaler.inverse_transform(y_pred)

#Calculo del error cuadrático medio
rmse_linear = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"\nError cuadrático medio del modelo Regresión lineal: {rmse_linear}")

#Calculo del error absoluto medio
mae_linear = metrics.mean_absolute_error(y_train_test, y_pred)
print(f"Error absoluto medio del modelo Regresión lineal: {mae_linear}")
np.random.seed(13)

#Usar predefined split para la validación cruzada
print("\nValidación cruzada con PredefinedSplit\n")

ps = PredefinedSplit(selector)

"""
print(ps.get_n_splits(X))

for train, valid in ps.split(X):
    print(f"Size of the train set: {train.shape}")
    print(f"Size of the valid set: {valid.shape}")
    print(f"Indices of the train set: {train}")
    print(f"Indices of the valid set: {valid}")
"""

#usar el predefined split para la validación cruzada
clf = LinearRegression()

np.random.seed(13)
scores = cross_val_score(clf, X_train_n, y_train_n, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo Regresión lineal calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

scores = cross_val_score(clf, X_train_n, y_train_n, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo Regresión lineal calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Probamos si es mejor que un dummy regressor linear

#Estrategy mean
regr_mean = DummyRegressor(strategy="mean")
regr_mean.fit(X_train_train_n, y_train_train_n)
rmse_mean = np.sqrt(metrics.mean_squared_error(y_train_test_n, regr_mean.predict(X_train_test_n)))

print("\nDummy Regressor:\n")
print(f"Error cuadrático medio del linear dummy (mean): {rmse_mean}")
print(f"RMSE ratio linear/dummy(mean): {rmse_mean/rmse_linear}")

#Estrategy median
regr_median = DummyRegressor(strategy="median")
regr_median.fit(X_train_train_n, y_train_train_n)
mae_dummy_linear = metrics.mean_absolute_error(y_train_test_n, regr_median.predict(X_train_test_n))

print(f"\nError absoluto medio del linear dummy (median): {mae_dummy_linear}")
print(f"MAE ratio linear/dummy(median): {mae_dummy_linear/mae_linear}")
print()



"""Ajuste de hiperparámetros"""
print("-"*60)
print("Evaluación de modelos con ajuste de hiperparámetros")
print("-"*60)

#KNN
print("\nKNN\n----")

#usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split
N_train = 7*365
N_test = 3*365

selector = [-1] * N_train + [0] * N_test
ps = PredefinedSplit(selector)

#Definimos los valores de los hiperparámetros que queremos probar

#n_neighbors
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8]

#weights
weights = ['uniform', 'distance']

#metric
metric = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

#leaf_size
#leaf_size = list(range(1,50))
leaf_size = [1, 2, 3, 5, 10, 30, 50, 100]

#Definimos el diccionario con los hiperparámetros
param_grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric, leaf_size=leaf_size)

#Definimos el modelo
model = KNeighborsRegressor()

#Definimos el grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

#Entrenamos el grid search
grid_result = grid.fit(X_train_n, y_train_n)

#Mejores hiperparámetros
print(f"\nMejores hiperparámetros: {grid_result.best_params_}")

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_test_n)

# Desnormalizar la predicción del modelo
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular la métrica de evaluación en la escala original
mae_knn_adjusted = metrics.mean_absolute_error(y_test, y_pred)
print(f"\nError absoluto medio del modelo KNN: {mae_knn_adjusted}")

#Mejor score
#mae_knn_adjusted = -grid_result.best_score_
#print(f"\nMejor score: {-grid_result.best_score_}")


#Arbol de decisión
print("\nÁrbol de decisión\n------------------")

#usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split
N_train = 7*365
N_test = 3*365

selector = [-1] * N_train + [0] * N_test
ps = PredefinedSplit(selector)

#Definimos los valores de los hiperparámetros que queremos probar

#criterion
#criterion = ['absolute_error', 'poisson', 'squared_error', 'friedman_mse']

#max_depth
#max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8]

#min_samples_split
min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]

#min_samples_leaf
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#Definimos el diccionario con los hiperparámetros
param_grid = dict(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
#param_grid = dict(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

#Definimos el modelo
model = tree.DecisionTreeRegressor()

#Definimos el grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

#Entrenamos el grid search
grid_result = grid.fit(X_train, y_train)

#Mejores hiperparámetros
print(f"\nMejores hiperparámetros: {grid_result.best_params_}")

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_train_test)

# Calcular la métrica de evaluación en la escala original
mae_tree_adjusted = metrics.mean_absolute_error(y_train_test, y_pred)
print(f"\nError absoluto medio del modelo Árbol de decisión: {mae_tree_adjusted}")

#Mejor score
#mae_tree_adjusted = -grid_result.best_score_
#print(f"\nMejor score: {-grid_result.best_score_}")


#Regresión lineal
print("\nRegresión lineal\n------------------")

#usaremos grid search para encontrar los mejores hiperparámetros haciendo antes predefined split
N_train = 7*365
N_test = 3*365

selector = [-1] * N_train + [0] * N_test
ps = PredefinedSplit(selector)

#Definimos los valores de los hiperparámetros que queremos probar

#fit_intercept
fit_intercept = [True, False]

#Definimos el diccionario con los hiperparámetros
param_grid = dict(fit_intercept=fit_intercept)

#Definimos el modelo
model = LinearRegression()

#Definimos el grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)

#Entrenamos el grid search
grid_result = grid.fit(X_train_n, y_train_n)

#Mejores hiperparámetros
print(f"\nMejores hiperparámetros: {grid_result.best_params_}")

# Obtener la mejor configuración de hiperparámetros y hacer una predicción en los datos de prueba normalizados
best_model = grid_result.best_estimator_
y_pred_n = best_model.predict(X_train_test_n)

# Desnormalizar la predicción del modelo
y_pred = scaler.inverse_transform(y_pred_n)

# Calcular la métrica de evaluación en la escala original
mae_linear_adjusted = metrics.mean_absolute_error(y_train_test, y_pred)
print(f"\nError absoluto medio del modelo linear: {mae_linear_adjusted}")

#Mejor score
#mae_linear_adjusted = -grid_result.best_score_
#print(f"\nMejor score: {-grid_result.best_score_}")


"""Comparación de modelos y resultados"""
print("-"*60)
print("Comparación de modelos y resultados")
print("-"*60)

#KNN
print("\nKNN\n----")

print("MAE sin ajustar:", mae_knn)
print("MAE ajustado:",mae_knn_adjusted)
print("MAE ratio knn/knn_adjusted:", mae_knn/mae_knn_adjusted)


#Arbol de decisión
print("\nÁrbol de decisión\n------------------")

print("MAE sin ajustar:", mae_tree)
print("MAE ajustado:",mae_tree_adjusted)
print("MAE ratio tree/tree_adjusted:", mae_tree/mae_tree_adjusted)
print("MAE ratio dummy/tree_adjusted:", mae_dummy_tree/mae_tree_adjusted)

#Regresión lineal
print("\nRegresión lineal\n------------------")

print("MAE sin ajustar:", mae_linear)
print("MAE ajustado:",mae_linear_adjusted)
print("MAE ratio linear/linear_adjusted:", mae_linear/mae_linear_adjusted)
print("(NO VALIDO POR NORMALIZACIÓN) MAE ratio dummy/linear_adjusted:", mae_dummy_linear/mae_linear_adjusted)

print()