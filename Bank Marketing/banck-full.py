from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.arbol import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, llamadaScore
from sklearn.model_selection import KFold, train_test_split
from matplotlib.pyplot import axis
from sklearn.svm import SVC
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("bank-full.csv")

data.drop(
    [
        "duracion",
        "pdays",
        "previous",
        "month",
        "balance",
        "trabajo",
        "marital",
        "default",
        "dia",
        "contact",
        "campaña",
        "poutcome",
    ],
    axis=1,
    inplace=True,
)

data.dropna(axis=0, inplace=True, how="any")

data.loan.replace(["si", "no"], [1, 0], inplace=True)
data.housing.replace(["si", "no"], [1, 0], inplace=True)
data.y.replace(["si", "no"], [1, 0], inplace=True)
data.education.replace(
    ["unknown", "secondary", "primary", "tertiary"], [0, 1, 2, 3], inplace=True
)

print(data.loan.value_counts())

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)


num_of_rows = data.shape[0]
train_size = int(num_of_rows * 0.8)
test_size = num_of_rows - train_size

train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

x = np.array(train_data.drop(["y"], axis=1))
y = np.array(train_data["y"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_pruebas = np.array(test_data.drop(["y"], axis=1))
y_pruebas = np.array(test_data["y"])



kfold = KFold(n_splits=10)
acc_scores_train_train = []
entrenamientoAcc = []
for entreanmiento, testeo in kfold.split(x, y):
    x_train, x_test = x[entreanmiento], x[testeo]
    y_train, y_test = y[entreanmiento], y[testeo]
    log_reg.fit(x_train, y_train)
    acc_scores_train_train.append(log_reg.score(x_train, y_train))
    entrenamientoAcc.append(log_reg.score(x_test, y_test))

y_pred = log_reg.predict(x_pruebas)
entrenamientoAccu = np.mean(acc_scores_train_train)
pruebasAccu = np.mean(entrenamientoAcc)
validacionAccu = log_reg.score(x_pruebas, y_pruebas)
recall = llamadaScore(y_pruebas, y_pred)
precision = precision_score(y_pruebas, y_pred)
f1 = f1_score(y_pruebas, y_pred)
matrix = confusion_matrix(y_pruebas, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
print("------------------ Logistic Regression ------------------")
print("Accuracy entreanmiento-entreanmiento : ", np.mean(acc_scores_train_train))
print("Accuracy entreanmiento-testeo  : ", np.mean(entrenamientoAcc))
print("Train accuracy       : ", entrenamientoAccu)
print("Test accuracy        : ", pruebasAccu)
print("Validation accuracy  : ", validacionAccu)
print("Recall               : ", recall)
print("Precision            : ", precision)
print("F1 score             : ", f1)
print("Real                 : ", y_pruebas)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
entrenamientoAcc = []
for entreanmiento, testeo in kfold.split(x):
    x_train, x_test = x[entreanmiento], x[testeo]
    y_train, y_test = y[entreanmiento], y[testeo]
    svc.fit(x_train, y_train)
    acc_scores_train_train.append(svc.score(x_train, y_train))
    entrenamientoAcc.append(svc.score(x_test, y_test))

y_pred = svc.predict(x_pruebas)
entrenamientoAccu = np.mean(acc_scores_train_train)
pruebasAccu = np.mean(entrenamientoAcc)
validacionAccu = svc.score(x_pruebas, y_pruebas)
recall = llamadaScore(y_pruebas, y_pred)
precision = precision_score(y_pruebas, y_pred)
f1 = f1_score(y_pruebas, y_pred)
matrix = confusion_matrix(y_pruebas, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - SVC")
plt.show()
print("------------------ Support Vector Machine ------------------")
print("Accuracy entreanmiento-entreanmiento: ", np.mean(acc_scores_train_train))
print("Accuracy entreanmiento-testeo : ", np.mean(entrenamientoAcc))
print("Train accuracy      : ", entrenamientoAccu)
print("Test accuracy       : ", pruebasAccu)
print("Validation accuracy : ", validacionAccu)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_pruebas)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
entrenamientoAcc = []
for entreanmiento, testeo in kfold.split(x):
    x_train, x_test = x[entreanmiento], x[testeo]
    y_train, y_test = y[entreanmiento], y[testeo]
    arbol.fit(x_train, y_train)
    acc_scores_train_train.append(arbol.score(x_train, y_train))
    entrenamientoAcc.append(arbol.score(x_test, y_test))

y_pred = arbol.predict(x_pruebas)
entrenamientoAccu = np.mean(acc_scores_train_train)
pruebasAccu = np.mean(entrenamientoAcc)
validacionAccu = arbol.score(x_pruebas, y_pruebas)
recall = llamadaScore(y_pruebas, y_pred)
precision = precision_score(y_pruebas, y_pred)
f1 = f1_score(y_pruebas, y_pred)
matrix = confusion_matrix(y_pruebas, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix (Decision Tree)")
plt.show()
print("------------------ Decision Tree ------------------")
print("Accuracy entreanmiento-entreanmiento: ", np.mean(acc_scores_train_train))
print("Accuracy entreanmiento-testeo : ", np.mean(entrenamientoAcc))
print("Train accuracy      : ", entrenamientoAccu)
print("Test accuracy       : ", pruebasAccu)
print("Validation accuracy : ", validacionAccu)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_pruebas)
print("Predicted            : ", y_pred)


kfold = KFold(n_splits=10)
acc_scores_train_train = []
entrenamientoAcc = []
for entreanmiento, testeo in kfold.split(x):
    x_train, x_test = x[entreanmiento], x[testeo]
    y_train, y_test = y[entreanmiento], y[testeo]
    clasificadorKN.fit(x_train, y_train)
    acc_scores_train_train.append(clasificadorKN.score(x_train, y_train))
    entrenamientoAcc.append(clasificadorKN.score(x_test, y_test))

y_pred = clasificadorKN.predict(x_pruebas)
entrenamientoAccu = np.mean(acc_scores_train_train)
pruebasAccu = np.mean(entrenamientoAcc)
validacionAccu = clasificadorKN.score(x_pruebas, y_pruebas)
recall = llamadaScore(y_pruebas, y_pred)
precision = precision_score(y_pruebas, y_pred)
f1 = f1_score(y_pruebas, y_pred)
matrix = confusion_matrix(y_pruebas, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - KNN")
plt.show()
print("------------------ K-Nearest Neighbors ------------------")
print("Accuracy entreanmiento-entreanmiento: ", np.mean(acc_scores_train_train))
print("Accuracy entreanmiento-testeo : ", np.mean(entrenamientoAcc))
print("Train accuracy      : ", entrenamientoAccu)
print("Test accuracy       : ", pruebasAccu)
print("Validation accuracy : ", validacionAccu)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_pruebas)
print("Predicted            : ", y_pred)

kfold = KFold(n_splits=10)
acc_scores_train_train = []
entrenamientoAcc = []
for entreanmiento, testeo in kfold.split(x):
    x_train, x_test = x[entreanmiento], x[testeo]
    y_train, y_test = y[entreanmiento], y[testeo]
    aletorioForest.fit(x_train, y_train)
    acc_scores_train_train.append(aletorioForest.score(x_train, y_train))
    entrenamientoAcc.append(aletorioForest.score(x_test, y_test))

y_pred = aletorioForest.predict(x_pruebas)
entrenamientoAccu = np.mean(acc_scores_train_train)
pruebasAccu = np.mean(entrenamientoAcc)
validacionAccu = aletorioForest.score(x_pruebas, y_pruebas)
recall = llamadaScore(y_pruebas, y_pred)
precision = precision_score(y_pruebas, y_pred)
f1 = f1_score(y_pruebas, y_pred)
matrix = confusion_matrix(y_pruebas, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(matrix)
plt.title("Confusion Matrix - Random Forest")
plt.show()
print("------------------ Random Forest ------------------")
print("Accuracy entreanmiento-entreanmiento: ", np.mean(acc_scores_train_train))
print("Accuracy entreanmiento-testeo : ", np.mean(entrenamientoAcc))
print("Train accuracy      : ", entrenamientoAccu)
print("Test accuracy       : ", pruebasAccu)
print("Validation accuracy : ", validacionAccu)
print("Recall              : ", recall)
print("Precision           : ", precision)
print("F1 score            : ", f1)
print("Real                 : ", y_pruebas)
print("Predicted            : ", y_pred)



print("Models:")

log_reg = LogisticRegression(solver="lbfgs", max_iter=train_size)
log_reg.fit(x_train, y_train)
log_reg_accuracy = log_reg.score(x_test, y_test)
log_reg_accuracy_validation = log_reg.score(x_pruebas, y_pruebas)
print("------------------ Logistic Regression ------------------")
print("Accuracy           : ", log_reg_accuracy)
print("Accuracy validation: ", log_reg_accuracy_validation)

svc = SVC(gamma="auto")
svc.fit(x_train, y_train)
svc_accuracy = svc.score(x_test, y_test)
svc_accuracy_validation = svc.score(x_pruebas, y_pruebas)
print("------------------ Support Vector Machine ------------------")
print("Accuracy           : ", svc_accuracy)
print("Accuracy validation: ", svc_accuracy_validation)

arbol = DecisionTreeClassifier()
arbol.fit(x_train, y_train)
tree_accuracy = arbol.score(x_test, y_test)
tree_validation = arbol.score(x_pruebas, y_pruebas)
print("------------------ Decision Tree ------------------")
print("Accuracy           : ", tree_accuracy)
print("Accuracy validation: ", tree_validation)

clasificadorKN = KNeighborsClassifier()
clasificadorKN.fit(x_train, y_train)
kn_classifier_accuracy = clasificadorKN.score(x_test, y_test)
kn_classifier_validation = clasificadorKN.score(x_pruebas, y_pruebas)
print("------------------ K-Nearest Neighbors ------------------")
print("Accuracy           : ", kn_classifier_accuracy)
print("Accuracy validation: ", kn_classifier_validation)

aletorioForest = RandomForestClassifier()
aletorioForest.fit(x_train, y_train)
random_forest_accuracy = aletorioForest.score(x_test, y_test)
random_forest_validation = aletorioForest.score(x_pruebas, y_pruebas)
print("------------------ Random Forest ------------------")
print("Accuracy           : ", random_forest_accuracy)
print("Accuracy validation: ", random_forest_validation)

print("\n\n")
print("Cross Validation:")



