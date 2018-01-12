# nous allons sauver toutes nos données dans ce fichier

# nous faisons les importations necessaires
import pandas as pd
import os

def save_data(true_data, data_bs, data_rmft, data_rm, tau_rm, var_tau_rm):
    columns = ["X", "Y", "Z", "X_bs", "Y_bs", "X_rmft", "Y_rmft", "X_rm", "Y_rm", "tau_rm", "var_tau_rm"]
    data = pd.DataFrame(columns = columns)
    
    data["X"] = true_data["x"]
    data["Y"] = true_data["y"]
    data["Z"] = true_data["z"]
    
    data["X_bs"] = data_bs[0]
    data["Y_bs"] = data_bs[1]
    
    data["X_rmft"] = data_rmft[0]
    data["Y_rmft"] = data_rmft[1]
    
    data["X_rm"] = data_rm[0]
    data["Y_rm"] = data_rm[1]
    data["tau_rm"] = pd.Series(tau_rm)
    data["var_tau_rm"] = pd.Series(var_tau_rm)
    
    
    exist = True
    i = 1
    while(exist):
        if os.path.exists("data/Cadre Article/" + str(i) + ".csv"):
            i = i + 1
        else:
            exist = False
    
    data.to_csv("data/Cadre Article/" + str(i) + ".csv", sep = ',', index = False)