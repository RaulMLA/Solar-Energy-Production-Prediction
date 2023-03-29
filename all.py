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

from sklearn.model_selection import PredefinedSplit, cross_val_score

from sklearn.dummy import DummyRegressor

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

print()



#------------------------------------------------------------
"""Evaluación de modelos sin ajuste de hiperparámetros."""
#------------------------------------------------------------

print("-" * 60)
print("Evaluación de modelos sin ajuste de hiperparámetros.")
print("-" * 60)

#Volvemos a dividir los datos en entrenamiento y test porque la parttición test solo la usaremos en la evaluación final
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=3/10, random_state=13, shuffle=False)

#muestra el tamaño de los datos de entrenamiento y test nuevos
print("Datos train_train: " ,X_train_train.shape, y_train_train.shape)   # 2550 días -> 7 años.
print("Datos train_test: " ,X_train_test.shape, y_train_test.shape)     # 1100 días  ->  3 años.

#KNN
print("\nKNN\n----")

# Entrenamiento del modelo.
clf = KNeighborsRegressor()
np.random.seed(13)
clf.fit(X_train_train, y_train_train)
y_pred = clf.predict(X_train_test)

#Calculo del error cuadrático medio
rmse_knn = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"Error cuadrático medio del modelo KNN: {rmse_knn}")

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
scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo KNN calculado con validación cruzada: {-scores.mean()}")

np.random.seed(13)
scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo KNN calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Árbol de decisión
print("\nÁrbol de decisión\n------------------")

# Entrenamiento del modelo.
clf = tree.DecisionTreeRegressor()
np.random.seed(13)
clf.fit(X_train_train, y_train_train)

# Predicciones del conjunto de test.
y_pred = clf.predict(X_train_test)

#Calculo del error cuadrático medio
rmse_tree = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"Error cuadrático medio del modelo Árbol de decisión: {rmse_tree}")

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
scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo Árbol de decisión calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo Árbol de decisión calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Probamos si es mejor que un dummy regressor tree

#Estrategy mean
regr_mean = DummyRegressor(strategy="mean")
regr_mean.fit(X_train_train, y_train_train)
rmse_mean = np.sqrt(metrics.mean_squared_error(y_train_test, regr_mean.predict(X_train_test)))

print("\nDummy Regressor:\n")
print(f"Error cuadrático medio del arbol dummy (mean): {rmse_mean}")
print(f"RMSE ratio tree/dummy(mean): {rmse_tree/rmse_mean}")

#Estrategy median
regr_median = DummyRegressor(strategy="median")
regr_median.fit(X_train_train, y_train_train)
mae_median = metrics.mean_absolute_error(y_train_test, regr_median.predict(X_train_test))

print(f"\nError absoluto medio del arbol dummy (median): {mae_median}")
print(f"MAE ratio tree/dummy(median): {mae_tree/mae_median}")

#Regresión lineal
print("\nRegresión lineal\n------------------")

# Entrenamiento del modelo.
clf = LinearRegression()
np.random.seed(13)
clf.fit(X_train_train, y_train_train)

# Predicciones del conjunto de test.
y_pred = clf.predict(X_train_test)

#Calculo del error cuadrático medio
rmse_linear = np.sqrt(metrics.mean_squared_error(y_train_test, y_pred))
print(f"Error cuadrático medio del modelo Regresión lineal: {rmse_linear}")

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
scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_root_mean_squared_error')
print(f"Error cuadrático medio del modelo Regresión lineal calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

scores = cross_val_score(clf, X, y, cv=ps, scoring='neg_mean_absolute_error')
print(f"Error absoluto medio del modelo Regresión lineal calculado con validación cruzada: {-scores.mean()}")
np.random.seed(13)

#Probamos si es mejor que un dummy regressor linear

#Estrategy mean
regr_mean = DummyRegressor(strategy="mean")
regr_mean.fit(X_train_train, y_train_train)
rmse_mean = np.sqrt(metrics.mean_squared_error(y_train_test, regr_mean.predict(X_train_test)))

print("\nDummy Regressor:\n")
print(f"Error cuadrático medio del linear dummy (mean): {rmse_mean}")
print(f"RMSE ratio linear/dummy(mean): {rmse_linear/rmse_mean}")

#Estrategy median
regr_median = DummyRegressor(strategy="median")
regr_median.fit(X_train_train, y_train_train)
mae_median = metrics.mean_absolute_error(y_train_test, regr_median.predict(X_train_test))

print(f"\nError absoluto medio del linear dummy (median): {mae_median}")
print(f"MAE ratio linear/dummy(median): {mae_linear/mae_median}")
print()


"""Ajuste de hiperparámetros"""
print("-"*60)
print("Ajuste de hiperparámetros")
print("-"*60)

#KNN
print("\nKNN\n----")

#Arbol de decisión
print("\nÁrbol de decisión\n------------------")

#Regresión lineal
print("\nRegresión lineal\n------------------")


print()