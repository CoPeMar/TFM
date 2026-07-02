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
import Functions.Pressure_Correction as PC
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.integrate import quad

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

#No usamos la humedad
combined_df = combined_df.iloc[:,:302]

#El método integral requiere que calculemos el perfil de profundidad atmosférica, el cual
#se obtiene de la siguiente manera:
Mmol= 28.97 #Masa molar del aire seco, g/mol
R = 0.082 #Constante de los gases, atm*L/(mol*K)
P_mean = combined_df.iloc[:,:151].mean()/1013.25 #Debe estar en atm
T_mean = combined_df.iloc[:,151:302].mean()+273.15 #Debe estar en K
rho = P_mean*Mmol/(T_mean*R) #Densidad del aire, g/L = kg/m3

#Realizamos un ajuste para poder integrar
m,b=np.polyfit(rho.index,np.log(rho.values),1)
A0 = np.exp(b)
k0=-m
popt,pcov=curve_fit(lambda t,a,b: a*np.exp(-b*t),rho.index,rho.values,p0=[A0,k0])
#Visualizamos el ajuste
#plt.plot(rho.index,popt[0]*np.exp(-rho.index*popt[1]))
#plt.plot(rho)
#Bastante bueno. Ahora hacemos la integración para obtener los valores de la profundidad
#atmosférica
x=pd.Series(index=rho.index) #Profundidad atmosférica
for i in rho.index:
    I = quad(lambda t,a,b: a*np.exp(-b*t),i,np.inf,args=(popt[0],popt[1]))
    x[i] = I[0]/10 #Pasamos a g/cm2
    
#Cambio de planes. Ya que no tengo datos concretos del detector de Izaña, paso a usar el
#método MSS (Mass Weighted)
Tmss = pd.Series(index=combined_df.index,data=float(0))
for j in range(len(combined_df.index)):
    for i in range(1,len(x.index)):
        if i == 1:
            Tmss.iloc[j] = 0.06*(combined_df.iloc[j,i+150]+273.15)
        else:
            Tmss.iloc[j] += ((x[x.index[i-1]]-x[x.index[i]])/x[2364])*(combined_df.iloc[j,i+150]+273.15)

DTmss = pd.Series(index=combined_df.index,data=float(0))
for i in range(len(combined_df.index)):
    DTmss.iloc[i] = Tmss.iloc[i]-Tmss.mean()
    
#Hacemos una corrección previa en presión
IPC,var = PC.Pressure_Correction(combined_df,data_hour)

#Ahora, extraemos el coeficiente
IPC = (IPC-IPC.mean())/IPC.mean()
temp_coef = pd.DataFrame({"IPC": IPC, "DTmss": DTmss})
modelo = sm.OLS(temp_coef.IPC,sm.add_constant(temp_coef.DTmss)).fit()

#Esta será una corrección bastante mala. Pero hay que hacerla.
#Al contrario que en De Mendonça et al. (2016), el efecto de la Tmss parece ser positivo
#sobre el conteo de muones.
DIT = modelo.params["DTmss"]*DTmss
top_new = data_hour.top/(1+DIT)

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
print(f"Varianza eliminada: {statistics.variance(data_hour.top - top_new)/statistics.variance(data_hour.top)+var}")

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido"]
plt.figure(100)
Functions.plot_corr(False,False,corr,"N/A","top","MSS")

wper_top,wpsd_top,wper_corr,wpsd_corr = Functions.welch_comparison(data_hour.top,top_new,"MSS")
pper_top,ppsd_top,pper_corr,ppsd_corr = Functions.periodogram_comparison(data_hour.top,top_new,"MSS")

save_welch = pd.DataFrame({"Period": wper_corr,
                           "Welch_PSD": wpsd_corr})
save_periodogram = pd.DataFrame({"Period": pper_corr,
                                 "Periodogram_PSD": ppsd_corr})
save_welch.to_csv(f"Results/welch_MSS.csv", index=False)
save_periodogram.to_csv(f"Results/periodogram_MSS.csv", index=False)
corr.to_csv(f"Results/corr_MSS.csv")