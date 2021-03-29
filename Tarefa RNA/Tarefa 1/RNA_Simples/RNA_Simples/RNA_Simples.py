# importando a biblioteca NumPy
import numpy as np

# Função Cálculo da sigmóide
def sigmoid(value):
    return 1 / (1 + np.exp(- value))

# vetor com valores de entrada
v_entrada = np.array([0.8, -0.3])
print("Entrada: ", v_entrada)

b = 0.1
print("b: ", b)

#Pesos das ligações sinápticas
v_pesos = np.array([0.2, -0.1])
print("Pesos: ", v_pesos)

#Cálculo de combinação linear de entrada e pesos sinápticos
calculo = np.dot(v_entrada, v_pesos) + b
print("Combinação linear: ", calculo)

#Aplicando a função de ativação do neurônio
saida = sigmoid(calculo)
print("A saída da rede é: ", saida)