# Librerias usadas
import numpy as np
import matplotlib.pyplot as plt


# Función de activación (escalón unitario)
def activacion(x):
    return np.heaviside(x, 0)


def perceptron(patrones, clases, W, b, numEpoc):
    for epoch in range(numEpoc):
        # Calculo de las predicciones
        a = activacion(np.dot(patrones, W) + b)

        # Cálculo del error
        error = clases - a

        # Actualización de pesos y bias
        W += np.dot(patrones.T, error)
        b += np.sum(error, axis=0, keepdims=True)

    # Clasificación final
    afinal = np.dot(patrones, W) + b
    salida_final = activacion(afinal)

    print("Patrones de entrada:\n", patrones)
    print("Clases reales:\n", clases)
    print("Clases predichas:\n", salida_final)

    # Graficación de las líneas de decisión y los patrones de entrada
    plt.figure(figsize=(8, 6))

    # Graficar patrones de entrada
    for i, clase in enumerate(clases):
        if clase[0] == 0 and clase[1] == 0:  # Clase[0,0]
            plt.scatter(patrones[i, 0], patrones[i, 1], color='red', marker='o')  # Grafica patrones de clase [0,0]
        elif clase[0] == 0 and clase[1] == 1:  # Clase[0,1]
            plt.scatter(patrones[i, 0], patrones[i, 1], color='blue', marker='x')  # Grafica patrones de clase [0,1]
        elif clase[0] == 1 and clase[1] == 0:  # Clase[1,0]
            plt.scatter(patrones[i, 0], patrones[i, 1], color='green', marker='s')  # Grafica patrones de clase [1,0]
        elif clase[0] == 1 and clase[1] == 1:  # Clase[1,1]
            plt.scatter(patrones[i, 0], patrones[i, 1], color='purple', marker='^')  # Grafica patrones de clase [1,1]

    # Graficar líneas de decisión
    x_vals = np.linspace(-1, 5, 100)  # Crea arreglo de 100 valores equidistantes en el rango de -1 a 5
    for i in range(2):  # Dos iteraciones, una por cada linea de decision
        y_vals = -(W[0, i] * x_vals + b[0, i]) / W[1, i]  # Calculo de la linea de decision
        plt.plot(x_vals, y_vals, label=f'Línea de Decisión {i + 1}')  # Graficacion de la linea de decision

    plt.xlabel('Peso')
    plt.ylabel('Frecuencia de uso')
    plt.title('Líneas de Decisión y Patrones de Entrada')
    plt.legend()
    plt.grid()
    plt.show()


# Patrones
patrones = np.array([[0.7, 3], [1.5, 5], [2.0, 9], [0.9, 11], [4.2, 0], [2.2, 1], [3.6, 7], [4.5, 6]])

# Clases reales
clases = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])

# Inicialiazcion aleatoria de Pesos
W = np.random.uniform(size=(2, 2))

# Inicializacion aletoria de bias
b = np.random.uniform(size=(1, 2))

# Epocas de entrenamiento
numEpoc = 50

# Se llama a la funcion perceptron
perceptron(patrones, clases, W, b, numEpoc)
