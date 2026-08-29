import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as md
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime
from copy import copy
import statistics
import Functions.Functions as Functions
import Functions.Pressure_Correction as PC
import statsmodels.api as sm
from scipy.integrate import simpson

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

xd = md.DateFormatter("%Y-%m-%d")

#No usamos la humedad
combined_df = combined_df.iloc[:,:302]

#Fechas de eventos como FD, D o GLE. En parejas de comienzo y final.
event_dates = ["2023-11-25 00:00:00","2023-12-04 00:00:00",
              "2024-03-24 00:00:00","2024-04-01 00:00:00",
              "2024-05-09 00:00:00","2024-05-31 00:00:00",
              "2024-07-30 00:00:00","2024-08-06 00:00:00",
              "2024-08-10 00:00:00","2024-08-15 00:00:00",
              "2024-09-16 00:00:00","2024-09-22 00:00:00",
              "2024-10-05 00:00:00","2024-10-17 00:00:00",
              "2024-10-26 00:00:00","2024-11-08 00:00:00",
              "2024-11-27 00:00:00","2024-12-04 00:00:00",
              "2024-12-22 00:00:00","2025-01-12 00:00:00",
              "2025-01-30 00:00:00","2025-02-09 00:00:00"]

#Creamos rectángulos para marcar los eventos en las gráficas. 
#En este caso, se han marcado con un rectángulo rojo los eventos de tipo D,
#azul los eventos de tipo FD y verde los eventos de tipo FD + GLE.
event_rects = []
top_event_rects = []
for i in range(0, len(event_dates), 2):
    color = 'red' if i == 16 or i == 20 else 'green' if i == 6 or i == 8 or i == 18 else 'blue'
    label = 'D' if i == 16 or i == 20 else 'FD' if i == 6 or i == 8 or i == 18 else 'FD + GLE'
    event_rects.append(mpatches.Rectangle((md.date2num(datetime.strptime
                      (event_dates[i], "%Y-%m-%d %H:%M:%S")), 0.9), md.date2num
                      (datetime.strptime(event_dates[i+1], "%Y-%m-%d %H:%M:%S")) - 
                      md.date2num(datetime.strptime(event_dates[i], "%Y-%m-%d %H:%M:%S")), 
                      0.3, color=color, alpha=0.3, label=label))
    top_event_rects.append(mpatches.Rectangle((md.date2num(datetime.strptime
                          (event_dates[i], "%Y-%m-%d %H:%M:%S")), 18000), md.date2num
                          (datetime.strptime(event_dates[i+1], "%Y-%m-%d %H:%M:%S")) - 
                          md.date2num(datetime.strptime(event_dates[i], "%Y-%m-%d %H:%M:%S")), 
                          10000, color=color, alpha=0.3, label=label))

#El método de temperatura efectiva requiere que calculemos el perfil de profundidad 
#atmosférica, el cual se obtiene de la siguiente manera:
Mmol= 28.97 #Masa molar del aire seco, g/mol
R = 0.082 #Constante de los gases, atm*L/(mol*K)
P_mean = combined_df.iloc[:,:151].mean()/1013.25 #Debe estar en atm
T_mean = combined_df.iloc[:,151:302].mean()+273.15 #Debe estar en K
rho = P_mean*Mmol/(T_mean*R) #Densidad del aire, g/L = kg/m3

#Ahora hacemos la integración para obtener los valores de la profundidad
#atmosférica
x=pd.Series(index=rho.index) #Profundidad atmosférica
for i in range(len(rho.index)):
    I = simpson(rho[i:],x=rho.index[i:])
    x[rho.index[i]] = I/10 #Pasamos a g/cm2
    
#Ahora, creamos la función de pesos.
L_pi = 160 #Longitud de atenuación atmosférica piones, g/cm2
L_n = 120 #Longitud de atenuación atmosférica nucleones, g/cm2

w = (1/x)*(np.exp(-(x/L_pi))-np.exp(-(x/L_n)))
w.iloc[-1]=w.iloc[-2] #Para evitar nan
w.index = x

Teff = pd.Series(index=combined_df.index,data=float(0))
for i in range(len(combined_df.index)):
    T = pd.Series(combined_df.iloc[i,151:302]+273.15) #K
    T.index = x
    Teff.iloc[i] = (simpson(w*T,x=x)/simpson(w,x=x))
    
#Tenemos Teff. Ahora toca obtener el coeficiente de temperatura efectiva.
DTeff = pd.Series(index=combined_df.index,data=float(0))
for i in range(len(combined_df.index)):
    DTeff.iloc[i] = Teff.iloc[i]-Teff.mean()
    
#Corrección de presión
IPC,var = PC.Pressure_Correction(combined_df,data_hour)

#Ahora, extraemos el coeficiente
IPC = (IPC-IPC.mean())/IPC.mean()
temp_coef = pd.DataFrame({"IPC": IPC, "DTeff": DTeff})
modelo = sm.OLS(temp_coef.IPC,sm.add_constant(temp_coef.DTeff)).fit()

#Esta será una corrección bastante mala. Pero hay que hacerla.
#Al contrario que en De Mendonça et al. (2016), el efecto de la Tmss parece ser positivo
#sobre el conteo de muones.
DIT = modelo.params["DTeff"]*DTeff
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
first_legend = ax.legend(handles=legend_elements[:2], loc='lower left')
ax.add_artist(first_legend)
ax.legend(handles=legend_elements[2:],loc='upper right')
for i in top_event_rects:
    rect = copy(i)
    ax.add_patch(rect)
print(f"Varianza eliminada: {statistics.variance(data_hour.top - top_new)/statistics.variance(data_hour.top) + var}")

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido"]
plt.figure(100)
Functions.plot_corr(False,False,corr,"N/A","top","EFF")

wper_top,wpsd_top,wper_corr,wpsd_corr = Functions.welch_comparison(data_hour.top,top_new,"EFF")
pper_top,ppsd_top,pper_corr,ppsd_corr = Functions.periodogram_comparison(data_hour.top,top_new,"EFF")

save_welch = pd.DataFrame({"Period": wper_corr,
                           "Welch_PSD": wpsd_corr})
save_periodogram = pd.DataFrame({"Period": pper_corr,
                                 "Periodogram_PSD": ppsd_corr})
save_welch.to_csv(f"Results/welch_EFF.csv", index=False)
save_periodogram.to_csv(f"Results/periodogram_EFF.csv", index=False)
corr.to_csv(f"Results/corr_EFF.csv")

#Probemos también la corrección con pesos alternativos sugerida en el artículo
w_M = (x)*(np.exp(-(x/L_pi))-np.exp(-(x/L_n)))
w_M.index = x
Teff_M = pd.Series(index=combined_df.index,data=float(0))
for i in range(len(combined_df.index)):
    T = pd.Series(combined_df.iloc[i,151:302]+273.15) #K
    T.index = x
    Teff_M.iloc[i] = (simpson(w_M*T,x=x)/simpson(w_M,x=x))
    
#Tenemos Teff. Ahora toca obtener el coeficiente de temperatura efectiva.
DTeff_M = pd.Series(index=combined_df.index,data=float(0))
for i in range(len(combined_df.index)):
    DTeff_M.iloc[i] = Teff_M.iloc[i]-Teff_M.mean()

temp_coef = pd.DataFrame({"IPC": IPC, "DTeff_M": DTeff_M})
modelo_M = sm.OLS(temp_coef.IPC,sm.add_constant(temp_coef.DTeff_M)).fit()

#Esta será una corrección bastante mala. Pero hay que hacerla.
#Al contrario que en De Mendonça et al. (2016), el efecto de la Tmss parece ser positivo
#sobre el conteo de muones.
DIT_M = modelo_M.params["DTeff_M"]*DTeff_M
top_new_m = data_hour.top/(1+DIT_M)

fig = plt.figure(123)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación top original vs top corregido")

plt.plot(combined_df.index, data_hour["top"], color='r')
plt.plot(combined_df.index, top_new_m, color='b')
legend_elements = [Line2D([0], [0], color='r', label='top original'),
                   Line2D([0], [0], color='b', label='top corregido'),
                   mpatches.Patch(color='red', alpha=0.3, label='D'),
                   mpatches.Patch(color='blue', alpha=0.3, label='FD'),
                   mpatches.Patch(color='green', alpha=0.3, label='FD + GLE')]
first_legend = ax.legend(handles=legend_elements[:2], loc='lower left')
ax.add_artist(first_legend)
ax.legend(handles=legend_elements[2:],loc='upper right')
for i in top_event_rects:
    rect = copy(i)
    ax.add_patch(rect)
print(f"Varianza eliminada: {statistics.variance(data_hour.top - top_new_m)/statistics.variance(data_hour.top) + var}")

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new_m],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido"]
plt.figure()
Functions.plot_corr(False,False,corr,"N/A","top","EFF_M")

wper_top,wpsd_top,wper_corr,wpsd_corr = Functions.welch_comparison(data_hour.top,top_new_m,"EFF_M")
pper_top,ppsd_top,pper_corr,ppsd_corr = Functions.periodogram_comparison(data_hour.top,top_new_m,"EFF_M")

save_welch = pd.DataFrame({"Period": wper_corr,
                           "Welch_PSD": wpsd_corr})
save_periodogram = pd.DataFrame({"Period": pper_corr,
                                 "Periodogram_PSD": ppsd_corr})
save_welch.to_csv(f"Results/welch_EFF_M.csv", index=False)
save_periodogram.to_csv(f"Results/periodogram_EFF_M.csv", index=False)
corr.to_csv(f"Results/corr_EFF_M.csv")

day_fit = Functions.fit_2sin(data_hour.top)
day_fit_corr = Functions.fit_2sin(top_new)
day_fit_corr_m = Functions.fit_2sin(top_new_m)

plt.figure()
plt.hist(day_fit_corr.r_squared,bins=np.arange(0,1,0.05))
plt.vlines(0.9,0,100,"k",linestyles="dashed",label="r^2 = 0.9")
ax = plt.gca()
ax.set(ylabel="Frecuencia", xlabel="r^2", title="Valores de r^2 de cada día para datos corregidos")
ax.legend()

plt.figure()
plt.hist(day_fit_corr_m.r_squared,bins=np.arange(0,1,0.05))
plt.vlines(0.9,0,100,"k",linestyles="dashed",label="r^2 = 0.9")
ax = plt.gca()
ax.set(ylabel="Frecuencia", xlabel="r^2", title="Valores de r^2 de cada día para datos corregidos")
ax.legend()

print(f"Amplitud porcentual ciclo diurno: {day_fit_corr.B0_percent.mean():.2f}±{day_fit_corr.B0_percent.std(ddof=1):.2f}%")
print(f"Amplitud porcentual ciclo semidiurno: {day_fit_corr.B1_percent.mean():.2f}±{day_fit_corr.B1_percent.std(ddof=1):.2f}%")

print(f"Amplitud porcentual ciclo diurno M: {day_fit_corr_m.B0_percent.mean():.2f}±{day_fit_corr_m.B0_percent.std(ddof=1):.2f}%")
print(f"Amplitud porcentual ciclo semidiurno M: {day_fit_corr_m.B1_percent.mean():.2f}±{day_fit_corr_m.B1_percent.std(ddof=1):.2f}%")