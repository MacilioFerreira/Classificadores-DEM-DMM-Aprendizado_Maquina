# -*- coding: utf-8 -*-

import numpy as np

# Distância de Mahalanobis
def mahalanobis(valorTeste, centroide, matriz):
    normal = valorTeste
    transposto = np.transpose(valorTeste)
    v_normal = []
    v_transposto = []
    i = 0
    while i < len(valorTeste):
        v_normal.append(normal[i]-centroide[i])
        v_transposto.append(transposto[i]-centroide[i])
        i += 1

    intermediaria = np.dot(v_normal, matriz)
    d_Mahalanobis = np.dot(intermediaria, v_transposto)
    return d_Mahalanobis



# Quantidade de atributos classificadas corretamente
def positivo(classe, teste):
    cont = 0
    for valor in classe:
        if valor in teste:
            cont += 1
    return cont

def isOutra(classe1, classe2):
    cont = 0
    for valor in classe1:
        if (valor in classe2):
            cont += 1
    return cont


def classificadorDMM(data):
    # Gerando posições aleatórias..
    posicoes = np.random.permutation(len(data))

    # Embaralhando os dados..
    data_alterado = np.zeros(data.shape)

    for i in xrange(0, len(posicoes)):
        data_alterado[i] = data[posicoes[i]]

    # Separação entre treinamento e testes..
    t_treinamento = int((0.8) * (len(data)))
    t_testes = (len(data) - t_treinamento)

    # Conjunto de treinamento
    # treinamento = [data_alterado[i][0:6] for i in range(0, t_treinamento)]
    treinamento = np.zeros((t_treinamento, 7))
    for i in range(0, t_treinamento):
        linha = data_alterado[i].tolist()
        treinamento[i] = linha

    classe1 = []  # Hernia
    classe2 = []  # Spondylolisthesis
    classe3 = []  # Normal
    for linha in treinamento:
        if int(linha[-1]) == 1:
            classe1.append(linha[0:6])
        if int(linha[-1]) == 0:
            classe2.append(linha[0:6])
        if int(linha[-1]) == -1:
            classe3.append(linha[0:6])

    # Calcular centroide das classes
    centroide1 = np.mean(classe1, axis=0)
    centroide2 = np.mean(classe2, axis=0)
    centroide3 = np.mean(classe3, axis=0)

    # Calculando a matriz de covariância
    matriz = []
    for linha in treinamento:
        matriz.append(linha[0:6])

    matriz_covariancia = np.cov(matriz,None,0)

    # Usar a distância mahalanobis e classificar o conjunto de testes.
    # Conjunto de teste.
    teste = np.zeros((t_testes, 7))
    cont = 0
    for i in range(t_treinamento, (t_testes + t_treinamento)):
        linha = data_alterado[i].tolist()
        teste[cont] = linha  # [0:6]
        cont += 1

    # Classificando
    classe_1 = []
    classe_2 = []
    classe_3 = []
    for linha in teste:
        d1 = mahalanobis(linha[0:6], centroide1, matriz_covariancia)
        d2 = mahalanobis(linha[0:6], centroide2, matriz_covariancia)
        d3 = mahalanobis(linha[0:6], centroide3, matriz_covariancia)
        if (d1 < d2) and (d1 < d3):
            classe_1.append(linha.tolist())
        if (d2 < d1) and (d2 < d3):
            classe_2.append(linha.tolist())
        if (d3 < d1) and (d3 < d2):
            classe_3.append(linha.tolist())

    # Matriz de confusão
    t_c1 = []  # H
    t_c2 = []  # S
    t_c3 = []  # N
    # Colocando os atributos de teste em suas respectivas classes
    for linha in teste:
        if linha[-1] == 1:
                t_c1.append(linha.tolist())
        if linha[-1] == 0:
                t_c2.append(linha.tolist())
        if linha[-1] == -1:
                t_c3.append(linha.tolist())

    # Criando matriz de confusão
    matriz_cfs = []
    matriz_cfs.append([positivo(classe_1, t_c1), isOutra(classe_1, t_c2), isOutra(classe_1, t_c3)])
    matriz_cfs.append([isOutra(classe_2, t_c1), positivo(classe_2, t_c2), isOutra(classe_2, t_c3)])
    matriz_cfs.append([isOutra(classe_3, t_c1), isOutra(classe_3, t_c2), positivo(classe_3, t_c3)])

    return  matriz_cfs



