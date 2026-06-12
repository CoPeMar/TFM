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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statistics
import Functions.Functions as Functions

#Variables importantes
r_squared_threshold = 0.1 #Límite de R^2 para considerar la relación entre la componente 
                          #principal y el conteo de partículas relevante
explained_variance_threshold = 0.01 #Límite de varianza explicada para considerar la 
                                    #componente principal relevante
quiet_days_only = False #Si se quieren usar solo los días tranquilos para hacer la 
                        #comparación, o si se quieren usar todos los datos.
lowest_pressure_only = False #Si se quieren usar solo los datos de presión más bajos 
                             #para hacer la comparación, o si se quieren usar todos 
                             #los datos.
include_humidity = True #Si se quiere incluir la humedad en el PCA o no
Test_neural_network = False #Predicción de PCA usando los datos de muones sin corregir

header_1=np.append([2364],np.linspace(2500,77000,150))
header=np.append(header_1,[header_1,header_1])
combined_df = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined_df.csv",
                          parse_dates=True, index_col=0,) #Datos atmosféricos
combined_df.columns = header
data_hour = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/data_hour.csv",
                          parse_dates=True, index_col=0) #Datos de conteo de partículas

xd = md.DateFormatter("%Y-%m-%d %H:%M:%S")

if quiet_days_only:
    temp = pd.read_csv("C:/TFM_Data/Q_Days.txt")
    Q_Days = pd.DataFrame()
    Q_Days_2 = pd.DataFrame()
    for i in temp.columns[2:]:
        for j in temp.index:
            Q_Days.loc[j,i] = f"{temp.iloc[j,0]}-{temp.iloc[j,1]}-{temp.loc[j,i]}"
            
    for i in Q_Days.columns:
        for j in Q_Days.index:
            Q_Days_2.loc[j,i] = datetime.strptime(Q_Days.loc[j,i], "%Y-%m-%d")
            
    nn = Q_Days_2.to_numpy().flatten()
    nn = pd.to_datetime(nn, format="%Y-%m-%d")
    
    combined_df.drop(combined_df.index[~np.isin(combined_df.index.normalize(),nn)],
                      inplace=True,axis=0)
    data_hour.drop(data_hour.index[~np.isin(data_hour.index.normalize(),nn)],
                      inplace=True,axis=0)

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
for i in range(0, len(event_dates), 2):
    color = 'red' if i == 16 or i == 20 else 'green' if i == 6 or i == 8 or i == 18 else 'blue'
    label = 'D' if i == 16 or i == 20 else 'FD' if i == 6 or i == 8 or i == 18 else 'FD + GLE'
    event_rects.append(mpatches.Rectangle((md.date2num(datetime.strptime
                      (event_dates[i], "%Y-%m-%d %H:%M:%S")), 0.9), md.date2num
                      (datetime.strptime(event_dates[i+1], "%Y-%m-%d %H:%M:%S")) - 
                      md.date2num(datetime.strptime(event_dates[i], "%Y-%m-%d %H:%M:%S")), 
                      0.3, color=color, alpha=0.3, label=label))
   
#Hacemos PCA tras elegir y reescalar los datos
scaler = StandardScaler()
pca = PCA(n_components=30)

if not include_humidity:
    combined_df = combined_df.iloc[:,:302]
if lowest_pressure_only:
    combined_df = pd.concat([combined_df.iloc[:,0], combined_df.iloc[:,151:]], axis=1)
combined_df.dropna(inplace=True,axis=1)
combined_df_Postscale = scaler.fit_transform(combined_df)
pca_result = pca.fit_transform(combined_df_Postscale)

#Para buscar la relación entre las componentes principales y los datos originales.
loadings = pd.DataFrame(
    pca.components_,
    columns=combined_df.columns,
    index=[f"PC{i+1}" for i in range(pca.n_components_)]
)

#Distingue entre presión y temperatura para poder hacer la comparación posteriormente
if lowest_pressure_only:
    loadings.loc[-1] = ["P" if i < 1 else "T" if i < 152 else "R" for i in range(len(loadings.T))]
else:
    loadings.loc[-1] = ["P" if i < 151 else "T" if i < 302 else "R" for i in range(len(loadings.T))]

#Escogemos los componentes principales a usar según su varianza explicada y su relación
#con los datos de conteo.

index_top, top_new, scoef_top = Functions.correction(pca, 
                                                     explained_variance_threshold, 
                                                     r_squared_threshold, 
                                                     data_hour["top"], 
                                                     pca_result)    
fig = plt.figure(10)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación top original vs top corregido")
if quiet_days_only:
    sns.scatterplot(x=combined_df.index, y=data_hour["top"]/data_hour["top"].mean(), 
                    color='r',s=5)
    sns.scatterplot(x=combined_df.index, y=top_new/top_new.mean(), 
                    color='b',s=5)
else:
    plt.plot(combined_df.index, data_hour["top"]/data_hour["top"].mean(), color='r')
    plt.plot(combined_df.index, top_new/top_new.mean(), color='b')
plt.hlines(1, combined_df.index[0], combined_df.index[-1], colors="k", linestyles="dashed")
legend_elements = [Line2D([0], [0], color='r', label='top original'),
                   Line2D([0], [0], color='b', label='top corregido'),
                   Line2D([0], [0], color='k', linestyle='dashed', label='Media'),
                   mpatches.Patch(color='red', alpha=0.3, label='D'),
                   mpatches.Patch(color='blue', alpha=0.3, label='FD'),
                   mpatches.Patch(color='green', alpha=0.3, label='FD + GLE')]
ax.legend(handles=legend_elements, loc='lower left')
for i in event_rects:
    rect = copy(i)
    ax.add_patch(rect)
    
plt.ylim(0.8,1.22)

index_bottom, bottom_new, scoef_bottom = Functions.correction(pca,
                                                           explained_variance_threshold,
                                                           r_squared_threshold,
                                                           data_hour["bottom"],
                                                           pca_result)       
fig = plt.figure(11)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación bottom original vs bottom corregido")
if quiet_days_only:
    sns.scatterplot(x=combined_df.index, y=data_hour["bottom"]/data_hour["bottom"].mean(), 
                    color='r',s=5)
    sns.scatterplot(x=combined_df.index, y=bottom_new/bottom_new.mean(), 
                    color='b',s=5)
else:
    plt.plot(combined_df.index,data_hour["bottom"]/data_hour["bottom"].mean(),"r")
    plt.plot(combined_df.index,bottom_new/bottom_new.mean(),"b")
plt.hlines(1, combined_df.index[0], combined_df.index[-1], colors="k", linestyles="dashed")
legend_elements = [Line2D([0], [0], color='r', label='bottom original'),
                   Line2D([0], [0], color='b', label='bottom corregido'),
                   Line2D([0], [0], color='k', linestyle='dashed', label='Media'),
                   mpatches.Patch(color='red', alpha=0.3, label='D'),
                   mpatches.Patch(color='blue', alpha=0.3, label='FD'),
                   mpatches.Patch(color='green', alpha=0.3, label='FD + GLE')]
ax.legend(handles=legend_elements, loc='lower left')
for i in event_rects:
    rect = copy(i)
    ax.add_patch(rect)
    
plt.ylim(0.8,1.22)

index_coin8, coin8_new, scoef_coin8 = Functions.correction(pca,
                                                           explained_variance_threshold,
                                                           r_squared_threshold,
                                                           data_hour["c8"],
                                                           pca_result)      
fig = plt.figure(12)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación c8 original vs c8 corregido")
if quiet_days_only:
    sns.scatterplot(x=combined_df.index, y=data_hour["c8"]/data_hour["c8"].mean(), 
                    color='r', s=5)
    sns.scatterplot(x=combined_df.index, y=coin8_new/coin8_new.mean(), 
                    color='b', s=5)
else:
    plt.plot(combined_df.index,data_hour["c8"]/data_hour["c8"].mean(),"r")
    plt.plot(combined_df.index,coin8_new/coin8_new.mean(),"b")
plt.hlines(1, combined_df.index[0], combined_df.index[-1], colors="k", linestyles="dashed")
legend_elements = [Line2D([0], [0], color='r', label='c8 original'),
                   Line2D([0], [0], color='b', label='c8 corregido'),
                   Line2D([0], [0], color='k', linestyle='dashed', label='Media'),
                   mpatches.Patch(color='red', alpha=0.3, label='D'),
                   mpatches.Patch(color='blue', alpha=0.3, label='FD'),
                   mpatches.Patch(color='green', alpha=0.3, label='FD + GLE')]
ax.legend(handles=legend_elements, loc='lower left')
for i in event_rects:
    rect = copy(i)
    ax.add_patch(rect)
    
plt.ylim(0.8,1.22)

#Plot de los loadings de las PC usadas
Functions.plot_loadings(index_top,index_bottom,index_coin8,loadings)

#Creación de la matriz de correlación
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new,bottom_new,
                      coin8_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]+3:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original",
                "top_corregido","bottom_corregido","c8_corregido"]

plt.figure(100)
Functions.plot_corr(lowest_pressure_only,include_humidity,corr,r_squared_threshold,"top")
plt.figure(101)
Functions.plot_corr(lowest_pressure_only,include_humidity,corr,r_squared_threshold,"bottom")
plt.figure(102)
Functions.plot_corr(lowest_pressure_only,include_humidity,corr,r_squared_threshold,"c8")

if Test_neural_network:
    from sklearn.metrics import root_mean_squared_error
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import *
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    from scipy.stats import pearsonr
    
    from sklearn.metrics import accuracy_score
    
    #División de datos
    X = pca_result[:,index_top]
    y = data_hour["top"]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scalerY = MinMaxScaler()
    y_train = scalerY.fit_transform(y_train.values.reshape(-1,1))
    y_test = scalerY.transform(y_test.values.reshape(-1,1))
    
    #Parámetros de la red neuronal
    epochs = 100
    batch_size = 32
    dropout = 0.2
    optimizer = "adam"
    activation = "relu"
    
    #Red neuronal
    model = Sequential()
    model.add(keras.Input(shape=(len(index_top),)))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    #Entrenamiento de la red neuronal
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(x_test, y_test), verbose=0)
    fig = plt.figure(200)
    ax = plt.gca()
    Functions.plot_loss(history)
    
    model_predictions = model.predict(x_test).flatten()
    model_predictions = scalerY.inverse_transform(model_predictions.reshape(-1,1)).flatten()
    y_test = scalerY.inverse_transform(y_test).flatten()
    rmse = root_mean_squared_error(y_test, model_predictions)
    pearson_corr, _ = pearsonr(y_test, model_predictions)

    plot_df = pd.DataFrame({"Observed": y_test, "Predicted": model_predictions})

    Functions.plot_predict(plot_df["Observed"], plot_df["Predicted"])
    print(f"RMSE: {rmse}")
    print(f"Pearson correlation: {pearson_corr}")
    
print(f"Varianza eliminada: {statistics.variance(data_hour.top - top_new)/statistics.variance(data_hour.top)}")