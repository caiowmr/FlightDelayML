import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dados = pd.read_csv('voos.csv')  # lendo o arquivo csv
dadoslinhas = pd.read_csv('voos.csv', skiprows=1)  # lendo o arquivo csv sem a primeira linha
dados = dados.fillna(0)  # preenchendo os valores nulos com 0

numEpocas = 30  # Número de épocas.
q = len(dadoslinhas)  # Número de padrões.

# Atributos relevantes
dia_do_mes = np.array(dados["DAY_OF_MONTH"])
dia_da_semana = np.array(dados["DAY_OF_WEEK"])
companhia_aerea = np.array(dados["OP_CARRIER_AIRLINE_ID"])
hora_partida = np.array(dados["DEP_TIME"])
hora_chegada = np.array(dados["ARR_TIME"])
atraso_partida = np.array(dados["DEP_DEL15"])
distancia = np.array(dados["DISTANCE"])

# Variáveis de resposta
d = np.array(dados["ARR_DEL15"])

eta = 0.01  # Taxa de aprendizado (é interessante alterar para avaliar o comportamento)
m = 7  # Número de neurônios na camada de entrada (peso e PH)
N1 = 40  # Número de neurônios na camada escondida 1.
N2 = 16  # Número de neurônios na camada escondida 2.
N3 = 10  # Número de neurônios na camada escondida 3.
L = 1  # Número de neurônios na camada de saída. (-1 = Maçã E 1 = Laranja)

# Matriz de pesos para a primeira camada oculta
W1 = np.random.randn(N1, m + 1) / np.sqrt(m + 1)

# Matriz de pesos para a segunda camada oculta
W2 = np.random.randn(N2, N1 + 1) / np.sqrt(N1 + 1)

# Matriz de pesos para a terceira camada oculta
W3 = np.random.randn(N3, N2 + 1) / np.sqrt(N2 + 1)

# Matriz de pesos para a camada de saída
W4 = np.random.randn(L, N3 + 1) / np.sqrt(N3 + 1)

# Array para armazenar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas)  # Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# bias
bias = 1

# Entrada do Perceptron.
X = np.vstack((dia_do_mes, dia_da_semana, companhia_aerea, hora_partida, hora_chegada, atraso_partida, distancia))

# ===============================================================
# TREINAMENTO.
# ===============================================================

for i in range(numEpocas):
    for j in range(q):

        # Insere o bias no vetor de entrada (apresentação do padrão da rede)
        Xb = np.hstack((bias, X[:, j]))

        o1 = np.tanh(W1.dot(Xb))
        # Saída da Camada Escondida 1.
        o1b = np.insert(o1, 0, bias)

        o2 = np.tanh(W2.dot(o1b))
        # Saída da Camada Escondida 2.
        o2b = np.insert(o2, 0, bias)

        # Saída da Camada Escondida 3.
        o3 = np.tanh(W3.dot(o2b))
        o3b = np.insert(o3, 0, bias)

        # Aplicar função de ativação após adicionar o bias
        o3b[1:] = np.tanh(o3b[1:])

        # Saída da Camada de Saída.
        Y = np.tanh(W4.dot(o3b))

        # Cálculo do erro
        e = d[j] - Y

        # Erro Total.
        E[j] = (e.transpose().dot(e)) / 2

        # Error backpropagation.
        # Cálculo do gradiente na camada de saída.
        delta4 = e * (1 - Y * Y)  # Eq. (6)
        vdelta4 = W4.T.dot(delta4)  # Eq. (7)
        delta3 = (1 - o3b[1:]**2) * vdelta4[1:]  # Eq. (8)
        vdelta3 = W3.T.dot(delta3)  # Gradiente para a camada escondida 3
        delta2 = (1 - o2b[1:]**2) * vdelta3[1:]  # Gradiente para a camada escondida 2
        vdelta2 = W2.T.dot(delta2)  # Gradiente para a camada escondida 2
        delta1 = (1 - o1b[1:]**2) * vdelta2[1:]  # Gradiente para a camada de entrada

        # Atualização dos pesos.
        W1 = W1 + eta * (np.outer(delta1, Xb))
        W2 = W2 + eta * (np.outer(delta2, o1b))
        W3 = W3 + eta * (np.outer(delta3, o2b))
        W4 = W4 + eta * (np.outer(delta4, o3b))

    # Cálculo da média dos erros
    Etm[i] = E.mean()

plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.show()

# ===============================================================
# TESTE DA REDE.
# ===============================================================


Error_Test = np.zeros(q)  # Inicialize corretamente com o tamanho dos dados de teste

# TESTE DA REDE.
Error_Test = np.zeros(q)  # Inicialize corretamente com o tamanho dos dados de teste

# TESTE DA REDE.
Error_Test = np.zeros((q, 1))  # Inicialize corretamente com o tamanho dos dados de teste

for i in range(q):
    # Insere o bias no vetor de entrada
    Xb = np.hstack((bias, X[:, i]))

    # Saída da Camada Escondida 1.
    o1 = np.tanh(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)

    # Saída da Camada Escondida 2.
    o2 = np.tanh(W2.dot(o1b))
    o2b = np.insert(o2, 0, bias)

    # Saída da Camada Escondida 3.
    o3 = np.tanh(W3.dot(o2b))
    o3b = np.insert(o3, 0, bias)

    # Saída da Camada de Saída.
    Y = np.tanh(W4.dot(o3b))

    # Armazena a saída para análise ou impressão, se necessário
    print(Y)

    # Calcula o erro de teste
    Error_Test[i] = d[i] - Y[0]  # Extraia um único elemento do array

print(Error_Test)
print(np.round(Error_Test) - d)
