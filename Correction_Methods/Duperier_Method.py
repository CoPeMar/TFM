import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as md
from matplotlib.lines import Line2D
import numpy as np
import random
from pyparsing import col
import seaborn as sns
import dateutil
from datetime import datetime, timedelta
from copy import copy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statistics
import Functions.Functions as Functions
import statsmodels.api as sm

hay_wifi = False #Para mi conveniencia

header_1=np.append([2364],np.linspace(2500,77000,150))
header=np.append(header_1,[header_1,header_1])
if hay_wifi:
    combined_df = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined_df.csv",
                              parse_dates=True, index_col=0,) #Datos atmosféricos
else:
    combined_df = pd.read_csv("C:/TFM_Data/Datos_Juntos/combined_df.csv",
                              parse_dates=True, index_col=0,) #Datos atmosféricos
combined_df.columns = header
if hay_wifi:
    data_hour = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/data_hour.csv",
                            parse_dates=True, index_col=0) #Datos de conteo de partículas
else:
    data_hour = pd.read_csv("C:/TFM_Data/Datos_Juntos/data_hour.csv",
                            parse_dates=True, index_col=0) #Datos de conteo de partículas

xd = md.DateFormatter("%Y-%m-%d %H:%M:%S")
#El método Duperier utiliza tan solo la presión a nivel de suelo, la temperatura en la
#capa de máxima generación de muones y la altitud de dicha capa. Por lo tanto, primero
#debemos obtener la altitud a la que se encuentra la capa de los 100 mBar/hPa cada hora
#y después interpolar la temperatura a esa altitud.

#No usamos la humedad
combined_df = combined_df.iloc[:,:302]

#La siguiente función obtiene la altitud a la que se encuentra la capa de los 100 hPa
#cada hora.
def altura_para_presion_log(alturas, presiones, p_obj=100):
    alturas = alturas.astype(float)
    presiones = presiones
    
    idx=np.argsort(presiones, axis=1)
    hpa=[]

    for i in range(len(presiones)):
        hpa.append(np.interp(
            np.log(p_obj),
            np.log(presiones[i][idx[i]]),
            alturas[idx[i]]
        ))
    return hpa

altura_100hpa = altura_para_presion_log(np.array(combined_df.columns[:151]), 
                                        combined_df.iloc[:,:151].values, 
                                        p_obj=100)

#Y esta función interpola la temperatura a esa altitud cada hora.
def temp_interpolada(alturas, temperaturas, alt_obj):
    temperaturas = temperaturas
    alturas = alturas.astype(float)
    
    idx=np.argsort(temperaturas, axis=1)
    temp_interp = []
    
    for i in range(len(temperaturas)):
        temp_interp.append(np.interp(
            alt_obj[i],
            alturas[idx[i]],
            temperaturas[i][idx[i]]
        ))
    return temp_interp

temp_100hpa = temp_interpolada(np.array(combined_df.columns[[28,30]]),
                               combined_df.iloc[:,[179,181]].values,
                               altura_100hpa)

Datos_Duperier = pd.DataFrame({"Presion_suelo":combined_df.iloc[:,0],
                              "Altura_100hpa":[i/1000 for i in altura_100hpa],
                              "Temp_100hpa":temp_100hpa,})

scaler = StandardScaler(with_std = False)

Datos_Duperier = Datos_Duperier.join(data_hour["top"], how="inner")

Datos_Duperier = pd.DataFrame(scaler.fit_transform(Datos_Duperier),
                              index=Datos_Duperier.index, 
                              columns=Datos_Duperier.columns)

modelo = sm.OLS(Datos_Duperier["top"]/data_hour.top.mean(), sm.add_constant(Datos_Duperier[[
                                "Presion_suelo","Altura_100hpa","Temp_100hpa"]])).fit()

top_new = data_hour["top"]/((1+modelo.params["Presion_suelo"]*
                             (Datos_Duperier.Presion_suelo)+
                             modelo.params["Altura_100hpa"]*
                             (Datos_Duperier.Altura_100hpa)+
                             modelo.params["Temp_100hpa"]*
                             (Datos_Duperier.Temp_100hpa)))

fig = plt.figure(10)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación top original vs top corregido")

plt.plot(combined_df.index, data_hour["top"], color='r')
plt.plot(combined_df.index, top_new, color='b')
legend_elements = [Line2D([0], [0], color='r', label='top original'),
                   Line2D([0], [0], color='b', label='top corregido'),
                   mpatches.Patch(color='red', alpha=0.3, label='D'),
                   mpatches.Patch(color='blue', alpha=0.3, label='FD'),
                   mpatches.Patch(color='green', alpha=0.3, label='FD + GLE')]
ax.legend(handles=legend_elements, loc='lower left')
print(f"Varianza eliminada: {statistics.variance(data_hour.top - top_new)/statistics.variance(data_hour.top)}")

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido"]
plt.figure(100)
Functions.plot_corr(False,False,corr,"N/A","top","Duperier")
wper_top,wpsd_top,wper_corr,wpsd_corr = Functions.welch_comparison(data_hour.top,top_new,"Duperier")
pper_top,ppsd_top,pper_corr,ppsd_corr = Functions.periodogram_comparison(data_hour.top,top_new,"Duperier")

save_welch = pd.DataFrame({"Period": wper_corr,
                           "Welch_PSD": wpsd_corr})
save_periodogram = pd.DataFrame({"Period": pper_corr,
                                 "Periodogram_PSD": ppsd_corr})
save_welch.to_csv(f"Results/welch_Duperier.csv", index=False)
save_periodogram.to_csv(f"Results/periodogram_Duperier.csv", index=False)
corr.to_csv(f"Results/corr_Duperier.csv")