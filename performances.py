"""
Code pour obtenir les performances des filtres
"""

# nous faisons les importations necessaires
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from MSE import *
from save_data import *

#Change matplotlib fontsize on graphs
matplotlib.rcParams.update({'font.size': 10})

data = pd.read_csv("data/Cadre 1/4.csv")

#print(mean_distance_all_filters(data))
#
plt.figure()
plt.plot(data["X"], data["Y"], label="True trajectory")
plt.scatter(data["X_bs"], data["Y_bs"], label="Particle means")
plt.title("Bootstrap filter")
#plt.legend()

plt.figure()
plt.plot(data["X"], data["Y"], label="True trajectory")
plt.scatter(data["X_rmft"], data["Y_rmft"], label="Particle means")
plt.title("Resample move with fixes tau")
#plt.legend()

plt.figure()
plt.plot(data["X"], data["Y"], label="True trajectory")
plt.scatter(data["X_rm"], data["Y_rm"], label="True trajectory")
plt.title("Resample move")
#plt.legend()

plt.figure()
plt.plot(data["tau_rm"], label="tau_rm")
plt.title("Estimations of tau at each step")
plt.legend()

plt.figure()
plt.plot(data["var_tau_rm"], label="tau_rm")
plt.title("Variance of tau at each step")
plt.legend()

print(data["tau_rm"])

#liste_bs, liste_rmft, liste_rm = liste_distance_all_filters()
#print(np.mean(liste_bs))
#print(np.mean(liste_rmft))
#print(np.mean(liste_rm))
#print(np.std(liste_bs))
#print(np.std(liste_rmft))
#print(np.std(liste_rm))
##
#plot_boxplots(liste_bs, liste_rmft, liste_rm)

#liste_distance_bs, liste_distance_rmft, liste_distance_rm = sde_periode_graphe()
#
#plt.plot(liste_distance_bs, label = "BS")
#plt.plot(liste_distance_rmft, label = "RMFT")
#plt.plot(liste_distance_rm, label = "RM")
#plt.title("Moyenne des distances euclidiennes à la vraie trajectoire")
#plt.legend(loc = 2)
#plt.show()
#
#print(np.mean(liste_distance_bs))
#print(np.mean(liste_distance_rm))
#print(np.mean(liste_distance_rmft))