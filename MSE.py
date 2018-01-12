from sklearn.metrics import mean_squared_error as EQM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
    
def two_point_distance(X1, Y1, X2, Y2):
    return np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
    
def distance_filter(data, filter_est):
    N = len(data["X"])
    
    somme = 0    
    
    for i in range(0, N):
        somme = somme + two_point_distance(data["X"][i], data["Y"][i], data["X_" + filter_est][i], data["Y_" + filter_est][i])
        
    return somme
    
    
def distance_all_filters(data):
    distance_bs = distance_filter(data, "bs")
    distance_rmft = distance_filter(data, "rmft")
    distance_rm = distance_filter(data, "rm")
    
    return distance_bs, distance_rmft, distance_rm
    
def mean_distance_all_filters(data):
    N = len(data["X"])
    distance_bs = distance_filter(data, "bs")/N
    distance_rmft = distance_filter(data, "rmft")/N
    distance_rm = distance_filter(data, "rm")/N
    
    return distance_bs, distance_rmft, distance_rm
    
def liste_distance_all_filters():
    liste_bs = []
    liste_rmft = []
    liste_rm = []
    
    exist = True
    i = 1
    while(exist):
        if os.path.exists("data/Cadre Article/" + str(i) + ".csv"):
            data = pd.read_csv("data/Cadre Article/" + str(i) + ".csv")
            distances = mean_distance_all_filters(data)
            liste_bs.append(distances[0])
            liste_rmft.append(distances[1])
            liste_rm.append(distances[2])
            i = i + 1
        else:
            exist = False
            
    return liste_bs, liste_rmft, liste_rm
    
def plot_boxplots(liste_bs, liste_rmft, liste_rm):
    plt.figure()
    plt.boxplot(liste_bs)
    plt.title("Boxplot des résultats du Bootstrap filter")
    
    plt.figure()
    plt.boxplot(liste_rmft)
    plt.title("Boxplot des résultats de l'algorithme resample move avec tau fixé")
    
    plt.figure()
    plt.boxplot(liste_rm)
    plt.title("Boxplot des résultats de l'algorithme resample move ")
    
def distances_moyenne_point(point):
    distance_rm = 0
    distance_bs = 0
    distance_rmft = 0
    
    exist = True
    i = 1
    while(exist):
        if os.path.exists("data/Cadre Article/" + str(i) + ".csv"):
            data = pd.read_csv("data/Cadre Article/" + str(i) + ".csv")
            distance_bs = distance_bs + two_point_distance(data["X"][point], data["Y"][point], data["X_bs"][point], data["Y_bs"][point])
            distance_rmft = distance_rmft + two_point_distance(data["X"][point], data["Y"][point], data["X_rmft"][point], data["Y_rmft"][point])
            distance_rm = distance_rm + two_point_distance(data["X"][point], data["Y"][point], data["X_rm"][point], data["Y_rm"][point])
            i = i + 1
        else:
            exist = False
    
    N = i - 1
    return distance_bs/N, distance_rmft/N, distance_rm/N
    
def sde_periode_graphe():
    data = pd.read_csv("data/Cadre Article/1.csv")
    K = len(data["X"])
    
    liste_distance_bs = []
    liste_distance_rmft = []
    liste_distance_rm = []
    
    
    for i in range(0, K):
        distance_bs, distance_rmft, distance_rm = distances_moyenne_point(i)
        
        liste_distance_bs.append(distance_bs)
        liste_distance_rmft.append(distance_rmft)
        liste_distance_rm.append(distance_rm)
        
        print(i)
        
    return liste_distance_bs, liste_distance_rmft, liste_distance_rm