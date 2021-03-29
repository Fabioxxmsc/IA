# importando a biblioteca NumPy
import numpy as np

# Função Cálculo da sigmóide
def sigmoid(value):
    return 1 / (1 + np.exp(- value))

#Função derivada da sigmóide
def sigmoid_prime(value):
    return sigmoid(value) * (1 - sigmoid(value))

#Taxa de aprendizagem
learnrate = 0.5

# vetor com valores de entrada
v_entrada = np.array([1, 2, 3, 4])
print("Entrada: ", v_entrada)

y = np.array(0.5)

b = 0.5
print("b: ", b)

#Pesos das ligações sinápticas
v_pesos = np.array([0.5, -0.5, 0.3, 0.1])
print("Pesos: ", v_pesos)

#Cálculo de combinação linear de entrada e pesos sinápticos
calculo = np.dot(v_entrada, v_pesos) + b
print("Combinação linear: ", calculo)

#Aplicando a função de ativação do neurônio
saida = sigmoid(calculo)
print("A saída da rede é: ", saida)

#Erro Calcular de rede neural
erro = y - saida
print("Erro: ", erro)

#Termo de erro
termo_erro = erro * sigmoid_prime(calculo)
print("Termo de erro: ", termo_erro)

del_w = learnrate * termo_erro * v_entrada
print("del_w", del_w)

# Atualizar pesos (Inicio treinamento)
v_pesos += del_w
print("Pesos: ", v_pesos)

#Cálculo de combinação linear de entrada e pesos sinápticos
calculo = np.dot(v_entrada, v_pesos) + b

#Aplicando a função de ativação do neurônio
saida = sigmoid(calculo)
print("A saída da rede é: ", saida)