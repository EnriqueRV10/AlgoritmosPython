# Bibliotecas usadas
import random
import numpy as np
import matplotlib.pyplot as plt


class AlgoritmoPerceptron:
    def __init__(self, min_val, max_val, num_epoc, learning_rate, cantidadp, p1x1, p1y1, p1x2, p1y2, p2x1, p2y1, p2x2, p2y2):
        self.min_val = min_val
        self.max_val = max_val
        self.numEpoc = num_epoc
        self.learning_rate = learning_rate
        self.cantidadp = cantidadp
        self.p1 = [p1x1, p1y1]
        self.p2 = [p1x2, p1y2]
        self.p3 = [p2x1, p2y1]
        self.p4 = [p2x2, p2y2]
        self.W = None
        self.b = None
        self.patrones = None
        self.clases = None

    # Define el conjunto de entradas y de salidas esperadas
    def set_patrones_salidas(self, p_region1, p_region2):
        self.patrones = np.array(p_region1 + p_region2)
        # Calcular la cantidad de patrones en cada región
        num_patrones_region1 = len(p_region1)
        num_patrones_region2 = len(p_region2)

        # Crear las listas de clases (etiquetas) con valores 0.0 y 1.0
        self.clases = np.append(np.zeros(num_patrones_region1), np.ones(num_patrones_region2))

    # Genera una lista de puntos aleatorios dados dos puntos
    def generar_puntos_aleatorios_en_dos_regiones(self):
        def generar_puntos_en_region(punto1, punto2, cantidad):
            x1, y1 = min(punto1[0], punto2[0]), min(punto1[1], punto2[1])
            x2, y2 = max(punto1[0], punto2[0]), max(punto1[1], punto2[1])

            puntos_generados = []
            for _ in range(cantidad):
                x = random.uniform(x1, x2)
                y = random.uniform(y1, y2)
                puntos_generados.append((x, y))

            return puntos_generados

        puntos_region1 = generar_puntos_en_region(self.p1, self.p2, self.cantidadp)
        puntos_region2 = generar_puntos_en_region(self.p3, self.p4, self.cantidadp)

        return puntos_region1, puntos_region2

    # Funcion que asigna valores aleatorios a los pesos y el bias
    def set_w_b_ln(self):
        # Inicialiazcion aleatoria de Pesos en el rango de valores min_val a max_val
        self.W = np.random.uniform(self.min_val, self.max_val, size=2)
        # Inicializacion aletoria de bias
        self.b = np.random.uniform()

    # Funcion con el algoritmo perceptron
    def perceptron(self):
        for epoca in range(self.numEpoc):
            # Calculo de las predicciones
            a = np.heaviside(np.dot(self.patrones, self.W) + self.b, 0)

            # Cálculo del error
            error = self.clases - a

            # Actualización de pesos y bias
            self.W += np.dot(self.patrones.T, error)*self.learning_rate
            self.b += np.sum(error, axis=0, keepdims=True)
            AlgoritmoPerceptron.graficacion(self, epoca)

        # Clasificación final
        afinal = np.dot(self.patrones, self.W) + self.b
        salida_final = np.heaviside(afinal, 0)

        # Se muestran resultados en consola
        print("Patrones de entrada:\n", self.patrones)
        print("Clases reales:\n", self.clases)
        print("Clases predichas:\n", salida_final)
        print("learning_rate: \n", self.learning_rate)
        print("Error: \n", error)
        print("Pesos: \n", self.W)
        AlgoritmoPerceptron.graficacionFinal(self)

    def graficacionFinal(self):
        # Graficación de las líneas de decisión y los patrones de entrada
        plt.figure(figsize=(8, 6))

        # Graficar patrones de entrada
        for i, c in enumerate(self.clases):
            if c == 0:  # Clase 0
                plt.scatter(self.patrones[i, 0], self.patrones[i, 1], color='red', marker='o')  # Grafica patrones de clase 0
            elif c == 1:  # Clase 1
                plt.scatter(self.patrones[i, 0], self.patrones[i, 1], color='blue', marker='x')  # Grafica patrones de clase 1

        # Graficar líneas de decisión
        x_vals = np.linspace(-6, 5, 100)  # Crea un arreglo de 100 valores equidistantes en el rango de -6 a 5
        for i in range(1):  # Dos iteraciones, una por cada línea de decisión
            y_vals = -(self.W[0] * x_vals + self.b) / self.W[1]  # Calculo de la línea de decisión
            plt.plot(x_vals, y_vals, label=f'Línea de Decisión {i + 1}')  # Graficación de la línea de decisión

        # Asignacion de etiquetas
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Líneas de Decisión y Patrones de Entrada')
        plt.legend()
        plt.grid()
        plt.show()

    def graficacion(self, epoc):
        # Graficación de las líneas de decisión y los patrones de entrada
        plt.figure(figsize=(8, 6))

        # Graficar patrones de entrada
        for i, c in enumerate(self.clases):
            if c == 0:  # Clase 0
                plt.scatter(self.patrones[i, 0], self.patrones[i, 1], color='red', marker='o')  # Grafica patrones de clase 0
            elif c == 1:  # Clase 1
                plt.scatter(self.patrones[i, 0], self.patrones[i, 1], color='blue', marker='x')  # Grafica patrones de clase 1

        # Graficar líneas de decisión
        x_vals = np.linspace(-6, 5, 100)  # Crea un arreglo de 100 valores equidistantes en el rango de -6 a 5
        for i in range(1):  # Dos iteraciones, una por cada línea de decisión
            y_vals = -(self.W[0] * x_vals + self.b) / self.W[1]  # Calculo de la línea de decisión
            plt.plot(x_vals, y_vals, label=f'Línea de Decisión {i + 1}')  # Graficación de la línea de decisión

        # Asignacion de etiquetas
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Resultado de Epoca ' + str(epoc))
        plt.legend()
        plt.grid()
        plt.show()


per1 = AlgoritmoPerceptron(0.1, 0.6, 10, 1, 5, -6, 4, -2, 2, -1, 0, 4, -4)
r1, r2 = per1.generar_puntos_aleatorios_en_dos_regiones()
per1.set_patrones_salidas(r1, r2)
per1.set_w_b_ln()
per1.perceptron()
