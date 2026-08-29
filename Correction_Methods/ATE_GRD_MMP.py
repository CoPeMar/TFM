import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as md
from matplotlib.lines import Line2D
import numpy as np
from datetime import datetime
from copy import copy
from sklearn.preprocessing import StandardScaler
import statistics
import Functions.Functions as Functions
import Functions.Pressure_Correction as PC
import statsmodels.api as sm

hay_wifi = False #Para mi conveniencia
pressure_correction = False #Realizar una corrección previa en presión. Parece que empeora la corrección

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

#De Mendonça et al. (2016) plantea un grupo de correcciones que son similares a Duperier.
#En este código se aplican todas las correcciones posibles y sus combinaciones.

metodo = "GRD+MMP" #ATE, GRD, MMP, ATE+GRD, ATE+MMP, GRD+MMP, ATE+GRD+MMP

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

altura_100hpa = pd.DataFrame(altura_100hpa,index=combined_df.index)
temp_grd = pd.DataFrame(combined_df.iloc[:,151],index=combined_df.index)
temp_100hpa = pd.DataFrame(temp_100hpa,index=combined_df.index)
Presion_suelo = pd.DataFrame(combined_df.iloc[:,0],index=combined_df.index)

scaler=StandardScaler(with_std=False)
if pressure_correction:
    ori_muon_count, var = PC.Pressure_Correction(combined_df,data_hour)
else:
    ori_muon_count = data_hour.top

muon_data = pd.DataFrame(ori_muon_count,index=data_hour.index)
muon_data = pd.DataFrame(scaler.fit_transform(muon_data),index=data_hour.index)

#Determinamos el método a utilizar
match metodo:
    case "ATE":
        altura_100hpa = pd.DataFrame(scaler.fit_transform(altura_100hpa)/1000,
                                     index=altura_100hpa.index,columns=["altura"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                        sm.add_constant(pd.concat([altura_100hpa,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["altura"]*altura_100hpa.altura+
                                  modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
    
    case "GRD":
        temp_grd = pd.DataFrame(scaler.fit_transform(temp_grd),
                                index=temp_grd.index,columns=["temp_grd"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                        sm.add_constant(pd.concat([temp_grd,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["temp_grd"]*temp_grd.temp_grd+
                                  modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
        
    case "MMP":
        temp_100hpa = pd.DataFrame(scaler.fit_transform(temp_100hpa),
                                   index=temp_100hpa.index,columns=["temp_mmp"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                        sm.add_constant(pd.concat([temp_100hpa,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["temp_mmp"]*temp_100hpa.temp_mmp+
                                  modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
        
    case "ATE+GRD":
        altura_100hpa = pd.DataFrame(scaler.fit_transform(altura_100hpa)/1000,
                                     index=altura_100hpa.index,columns=["altura"])
        temp_grd = pd.DataFrame(scaler.fit_transform(temp_grd),
                                index=temp_grd.index,columns=["temp_grd"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                 sm.add_constant(pd.concat([altura_100hpa,temp_grd,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["altura"]*altura_100hpa.altura+
                                 modelo.params["temp_grd"]*temp_grd.temp_grd+
                                 modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
    case "ATE+MMP":
        altura_100hpa = pd.DataFrame(scaler.fit_transform(altura_100hpa)/1000,
                                     index=altura_100hpa.index,columns=["altura"])
        temp_100hpa = pd.DataFrame(scaler.fit_transform(temp_100hpa),
                                   index=temp_100hpa.index,columns=["temp_mmp"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                 sm.add_constant(pd.concat([altura_100hpa,temp_100hpa,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["altura"]*altura_100hpa.altura+
                                 modelo.params["temp_mmp"]*temp_100hpa.temp_mmp+
                                 modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
    case "GRD+MMP":
        temp_grd = pd.DataFrame(scaler.fit_transform(temp_grd),
                                index=temp_grd.index,columns=["temp_grd"])
        temp_100hpa = pd.DataFrame(scaler.fit_transform(temp_100hpa),
                                   index=temp_100hpa.index,columns=["temp_mmp"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                 sm.add_constant(pd.concat([temp_grd,temp_100hpa,Presion_suelo],axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["temp_grd"]*temp_grd.temp_grd+
                                 modelo.params["temp_mmp"]*temp_100hpa.temp_mmp+
                                 modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)
        
    case "ATE+GRD+MMP":
        altura_100hpa = pd.DataFrame(scaler.fit_transform(altura_100hpa)/1000,
                                     index=altura_100hpa.index,columns=["altura"])
        temp_grd = pd.DataFrame(scaler.fit_transform(temp_grd),
                                index=temp_grd.index,columns=["temp_grd"])
        temp_100hpa = pd.DataFrame(scaler.fit_transform(temp_100hpa),
                                   index=temp_100hpa.index,columns=["temp_mmp"])
        Presion_suelo = pd.DataFrame(scaler.fit_transform(Presion_suelo),
                                     index=Presion_suelo.index,columns=["Presion_suelo"])
        modelo = sm.OLS(muon_data/ori_muon_count.mean(),
                 sm.add_constant(pd.concat([altura_100hpa,temp_grd,temp_100hpa,Presion_suelo],
                                           axis=1))).fit()
        top_new = ori_muon_count/(1+modelo.params["temp_grd"]*temp_grd.temp_grd+
                                 modelo.params["temp_mmp"]*temp_100hpa.temp_mmp+
                                 modelo.params["altura"]*altura_100hpa.altura+
                                 modelo.params["Presion_suelo"]*Presion_suelo.Presion_suelo)       


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
print(f"Varianza eliminada: {statistics.variance(ori_muon_count - top_new)/statistics.variance(ori_muon_count)}")

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido"]
plt.figure(100)
Functions.plot_corr(False,False,corr,"N/A","top",metodo)

wper_top,wpsd_top,wper_corr,wpsd_corr = Functions.welch_comparison(ori_muon_count,top_new,metodo)
pper_top,ppsd_top,pper_corr,ppsd_corr = Functions.periodogram_comparison(ori_muon_count,top_new,metodo)

save_welch = pd.DataFrame({"Period": wper_corr,
                           "Welch_PSD": wpsd_corr})
save_periodogram = pd.DataFrame({"Period": pper_corr,
                                 "Periodogram_PSD": ppsd_corr})

if pressure_correction:
    save_welch.to_csv(f"Results/welch_{metodo}_pc.csv", index=False)
    save_periodogram.to_csv(f"Results/periodogram_{metodo}_pc.csv", index=False)
    corr.to_csv(f"Results/corr_{metodo}_pc.csv")
else:
    save_welch.to_csv(f"Results/welch_{metodo}.csv", index=False)
    save_periodogram.to_csv(f"Results/periodogram_{metodo}.csv", index=False)
    corr.to_csv(f"Results/corr_{metodo}.csv")
    
day_fit = Functions.fit_2sin(data_hour.top)
day_fit_corr = Functions.fit_2sin(top_new)

plt.figure()
plt.hist(day_fit_corr.r_squared,bins=np.arange(0,1,0.05))
plt.vlines(0.9,0,100,"k",linestyles="dashed",label="r^2 = 0.9")
ax = plt.gca()
ax.set(ylabel="Frecuencia", xlabel="r^2", title="Valores de r^2 de cada día para datos corregidos")
ax.legend()

print(f"Amplitud porcentual ciclo diurno: {day_fit_corr.B0_percent.mean():.2f}±{day_fit_corr.B0_percent.std(ddof=1):.2f}%")
print(f"Amplitud porcentual ciclo semidiurno: {day_fit_corr.B1_percent.mean():.2f}±{day_fit_corr.B1_percent.std(ddof=1):.2f}%")