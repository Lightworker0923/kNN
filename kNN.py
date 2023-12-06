import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Datos de ejemplo
x = [2, 3, 5, 7, 8, 10, 12, 15, 18, 20]
y = [15, 12, 18, 25, 22, 30, 28, 32, 35, 40]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Visualizar los datos
plt.scatter(x, y, c=classes)
plt.show()

# Crear un modelo KNN con K=1
data = list(zip(x, y))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)

# Clasificar un nuevo punto
new_x = 15
new_y = 36
new_point = [(new_x, new_y)]
prediction = knn.predict(new_point)

# Visualizar el resultado
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"Nuevo punto, clase: {prediction[0]}")
plt.show()

# Cambiar el valor de K a 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data, classes)
prediction = knn.predict(new_point)

# Visualizar el nuevo resultado
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"Nuevo punto, clase: {prediction[0]}")
plt.show()