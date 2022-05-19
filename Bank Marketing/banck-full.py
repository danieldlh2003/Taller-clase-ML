import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier

dataset = 'bank-full.csv'
data = pd.read_csv(dataset)

#eliminamos posibles datos nulos
data.dropna(axis=0,how='any', inplace=True)

#se convierten las edades en rangos
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)

#limpiamos y normalizamos la data
data.drop(['marital', 'month', 'balance', 'day', 'duration', 'campaign', 'pdays', 'poutcome'], axis=1, inplace=True)

#se rempalzan los datos por categorias numericas
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace=True)

data.job.replace(['admin.', 'unknown', 'unemployed', 'management', 
                  'housemaid', 'entrepreneur', 'student', 'blue-collar', 
                  'self-employed', 'retired', 'technician', 'services'], 
                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

data.loan.replace(['no', 'yes'], [0, 1], inplace=True)

data.default.replace(['no', 'yes'], [0, 1], inplace=True)

data.y.replace(['no', 'yes'], [0, 1], inplace=True)

data.housing.replace(['no', 'yes'], [0, 1], inplace=True)

data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)

#dividimos los datos en dos
data_train = data[:22000]
data_test = data[22000:]


x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y) # 0 no acepto, 1 si acepto
# Regresión Logística


# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

#VECINOS MAS CERCANOS

#eseleccionamos el modelo
kn = KNeighborsRegressor(n_neighbors=10)

#entrenamos el modelo
kn.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('VECINOS MAS CERCANOS')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {kn.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {kn.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {kn.score(x_test_out, y_test_out)}')


#RANDOM FOREST

#seleccionar un modelo
rf = RandomForestClassifier()

#entrenamos el modelo
rf.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {rf.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {rf.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {rf.score(x_test_out, y_test_out)}')




