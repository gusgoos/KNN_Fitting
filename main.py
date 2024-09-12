# Importar las bibliotecas necesarias de sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Cargar el dataset Wine desde sklearn
wine = load_wine()
X = wine.data
y = wine.target

# Dividir los datos en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=19)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=19)

# Estandarizar las características usando StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Implementar K-Nearest Neighbors usando el framework proporcionado por sklearn
k = 2  # Número de vecinos
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train, y_train)  # Ajustar el modelo con los datos de entrenamiento

# Predecir las etiquetas para el conjunto de validación
y_val_pred = knn.predict(X_val)

# Evaluar el modelo utilizando matriz de confusión y otros métricos en el conjunto de validación
print("Evaluación del modelo en el conjunto de validación:")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_val_pred)) # Calcular Matriz de Confusión

# Calcular Precisión, Recall y F1Score
precision = precision_score(y_val, y_val_pred, average='macro')
recall = recall_score(y_val, y_val_pred, average='macro')
f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Grid Search para ajuste de hiperparámetros en el conjunto de validación
param_grid = {'n_neighbors': np.arange(1, 50)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Imprimir el mejor valor de k encontrado por GridSearch
print(f"Mejor valor de k encontrado por GridSearch: {grid_search.best_params_['n_neighbors']}")

# Reajustar el modelo con el mejor valor de k encontrado
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
y_best_val_pred = best_knn.predict(X_val)

# Evaluar el nuevo modelo con el hiperparámetro k optimizado en el conjunto de validación
print("Evaluación del modelo después de Grid Search en el conjunto de validación:")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_best_val_pred))
precision = precision_score(y_val, y_best_val_pred, average='macro')
recall = recall_score(y_val, y_best_val_pred, average='macro')
f1 = f1_score(y_val, y_best_val_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Evaluar el modelo final en el conjunto de prueba
y_test_pred = best_knn.predict(X_test)

# Evaluar el modelo en el conjunto de prueba
print("Evaluación final del modelo en el conjunto de prueba:")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_test_pred))
precision = precision_score(y_test, y_test_pred, average='macro')
recall = recall_score(y_test, y_test_pred, average='macro')
f1 = f1_score(y_test, y_test_pred, average='macro')
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")





# Validación Cruzada usando K-Fold Cross Validation (cross_val_score)
# Calcular la precisión del modelo utilizando validación cruzada de 10 folds
scores = cross_val_score(best_knn, X, y, cv=10, scoring='accuracy')
print(f"Precisión del modelo usando validación cruzada de 10 folds: {np.mean(scores):.2f}")

# Determinar Bias (underfitting) o Varianza (overfitting)
# Analizar error entrenamiento vs error prueba
train_error = 1 - best_knn.score(X_train, y_train)
test_error = 1 - best_knn.score(X_test, y_test)
print(f"Error de entrenamiento: {train_error:.2f}")
print(f"Error de prueba: {test_error:.2f}")

# Validar con Learning Curve
# Generar la curva de aprendizaje para analizar la evolución del rendimiento del modelo
train_sizes, train_scores, test_scores = learning_curve(best_knn, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
# Calcular los promedios y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
# Graficar la curva de aprendizaje
plt.figure()
plt.title("Curva de Aprendizaje: K-Nearest Neighbors")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión")
plt.grid()
# Graficar las puntuaciones medias
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntuación de entrenamiento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Puntuación de validación cruzada")
plt.legend(loc="best")
plt.show()
