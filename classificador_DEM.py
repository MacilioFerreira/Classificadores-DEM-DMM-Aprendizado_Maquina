# -*- coding: utf-8 -*-
import numpy as np
import math

# Distância Euclediana
def euclidiana(valorTeste, centroide):
    distEucli = 0
    i = 0
    while i < len(valorTeste):
        distEucli += math.pow((np.abs(valorTeste[i] - centroide[i])), 2)
        i += 1

    return math.sqrt(distEucli)

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


def classificadorDEM(data):

    # Normalizar os dados
    # Médias
    medias = []
    medias.append(np.mean([data[a][0] for a in range(len(data))], 0))
    medias.append(np.mean([data[a][1] for a in range(len(data))], 0))
    medias.append(np.mean([data[a][2] for a in range(len(data))], 0))
    medias.append(np.mean([data[a][3] for a in range(len(data))], 0))
    medias.append(np.mean([data[a][4] for a in range(len(data))], 0))
    medias.append(np.mean([data[a][5] for a in range(len(data))], 0))

    # Desvio Padrão
    desvioGeral = []
    desvioGeral.append(np.std([data[a][0] for a in range(len(data))], 0))
    desvioGeral.append(np.std([data[a][1] for a in range(len(data))], 0))
    desvioGeral.append(np.std([data[a][2] for a in range(len(data))], 0))
    desvioGeral.append(np.std([data[a][3] for a in range(len(data))], 0))
    desvioGeral.append(np.std([data[a][4] for a in range(len(data))], 0))
    desvioGeral.append(np.std([data[a][5] for a in range(len(data))], 0))

    # Normalizar, cada atributo recebe a diferença dele para a sua média sobre o desvio padrão
    for coluna in data:
        coluna[0] = np.abs((coluna[0] - medias[0]) / desvioGeral[0])
        coluna[1] = np.abs((coluna[1] - medias[1]) / desvioGeral[1])
        coluna[2] = np.abs((coluna[2] - medias[2]) / desvioGeral[2])
        coluna[3] = np.abs((coluna[3] - medias[3]) / desvioGeral[3])
        coluna[4] = np.abs((coluna[4] - medias[4]) / desvioGeral[4])
        coluna[5] = np.abs((coluna[5] - medias[5]) / desvioGeral[5])

    # Gerando posições aleatórias..
    posicoes = np.random.permutation(len(data))

    # Embaralhando os dados..
    data_alterado = np.zeros(data.shape)

    for i in xrange(0, len(posicoes)):
        data_alterado[i] = data[posicoes[i]]

    # Separação entre treinamento e testes..
    t_treinamento = int((0.8)*(len(data)))
    t_testes = (len(data) - t_treinamento)

    # Dados normalizados, encontrar classes
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

    # Usar a distância euclidiana e classificar o conjunto de testes.
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
        d1 = euclidiana(linha[0:6], centroide1)
        d2 = euclidiana(linha[0:6], centroide2)
        d3 = euclidiana(linha[0:6], centroide3)
        if (d1 < d2) and (d1 < d3):
            classe_1.append(linha.tolist())
        if (d2 < d1) and (d2 < d3):
            classe_2.append(linha.tolist())
        if (d3 < d1) and (d3 < d2):
            classe_3.append(linha.tolist())  # Inserindo a linha completa, facilita o teste

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

