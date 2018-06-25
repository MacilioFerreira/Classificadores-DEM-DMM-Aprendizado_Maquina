#-*- coding: utf-8 -*-
import numpy as np
import classificador_DEM as DEM
import classificador_DMM as DMM
import tabulate as tabela

def gerarRelatorio(data, tipo):
    # Quantidade de iterações
    quantidade = 50

    dem_geral = []
    dem_hernia = []
    dem_spondy = []
    dem_normal = []
    matriz_confusao = np.zeros((3, 3))  # Quantidade de classes
    matriz_confusao = matriz_confusao.tolist()

    j = 0
    while j < quantidade:
        # Resultado classifição
        if tipo == 1:
            matriz = DEM.classificadorDEM(data)
        else:
            matriz = DMM.classificadorDMM(data)

        # Distribuindo resultados..
        i = 0
        while i < len(matriz):
            if i == 0:
                dem_hernia.append(matriz[i][0]/float(quantidade))
                matriz_confusao[i][0] = matriz_confusao[i][0] + matriz[i][0]
                matriz_confusao[i][1] = matriz_confusao[i][1] + matriz[i][1]
                matriz_confusao[i][2] = matriz_confusao[i][2] + matriz[i][2]
            if i == 1:
                dem_spondy.append(matriz[i][0]/float(quantidade))
                matriz_confusao[i][0] = matriz_confusao[i][0] + matriz[i][0]
                matriz_confusao[i][1] = matriz_confusao[i][1] + matriz[i][1]
                matriz_confusao[i][2] = matriz_confusao[i][2] + matriz[i][2]
            if i == 2:
                dem_normal.append(matriz[i][0]/float(quantidade))
                matriz_confusao[i][0] = matriz_confusao[i][0] + matriz[i][0]
                matriz_confusao[i][1] = matriz_confusao[i][1] + matriz[i][1]
                matriz_confusao[i][2] = matriz_confusao[i][2] + matriz[i][2]
            dem_geral.append(matriz[i][0]/float(quantidade))
            i += 1
        j += 1

    print "\nTaxas de Acertos Geral: \n"
    print "Média: " + str(round(np.mean(dem_geral), 3)) + "\nMínima: " + str(min(dem_geral)) + "\nMáxima: " + str(
        max(dem_geral))
    print "\nTaxas de Acertos por Classe: \n"
    print "Hernia: \nMédia: " + str(np.mean(dem_hernia)) + "\nMínima: " + str(min(dem_hernia)) + "\nMáxima: " + str(
        max(dem_hernia))
    print "\nSpondylolisthesis: \nMédia: " + str(np.mean(dem_spondy)) + "\nMínima: " + str(
        min(dem_spondy)) + "\nMáxima: " + str(max(dem_spondy))
    print "\nNormal: \nMédia: " + str(np.mean(dem_normal)) + "\nMínima: " + str(min(dem_normal)) + "\nMáxima: " + str(
        max(dem_normal))

    # Separação entre treinamento e testes..
    t_treinamento = int((0.8) * (len(data)))
    t_testes = (len(data) - t_treinamento)

    i = 0
    while i < len(matriz_confusao):
        j = 0
        while j < len(matriz_confusao):
            matriz_confusao[i][j] = round(matriz_confusao[i][j] / float(t_testes * 50), 4)
            j += 1
        i += 1

    matriz_cfs = [["Hernia"] + matriz_confusao[0], ["Spondylolisthesis"] + matriz_confusao[1],
                  ["Normal"] + matriz_confusao[2]]

    print "\n                                       Matriz de Confusão Geral: Classificador de Distância Euclidiana \n"
    print(tabela.tabulate(matriz_cfs, ["Hernia", "Spondylolisthesis", "Normal"], tablefmt="grid"))


# Lendo o arquivo
data = np.genfromtxt("column_3C.dat", delimiter=",") # Mesmo conjunto de dados, porém este especifica as classes

print "\n                                   Classificador 01: Distância Euclidiana                      \n"
gerarRelatorio(data, 1)

print "\n                                   Classificador 02: Distância de Mahalanobis                      \n"
gerarRelatorio(data, 2)

