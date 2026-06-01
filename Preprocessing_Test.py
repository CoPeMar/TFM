import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as md
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import dateutil
from datetime import datetime
from copy import copy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Variables importantes
r_squared_threshold = 0.01 #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
explained_variance_threshold = 0.01 #Límite de varianza explicada para considerar la componente principal relevante
quiet_days_only = False #Si se quieren usar solo los días tranquilos para hacer la comparación, o si se quieren usar todos los datos.
lowest_pressure_only = False #Si se quieren usar solo los datos de presión más bajos para hacer la comparación, o si se quieren usar todos los datos.
Test_neural_network = True #Predicción de PCA usando los datos de muones sin corregir

#Funciones útiles
def plot_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()
  
def plot_predict(observed, predicted):
  plt.figure(figsize=(20, 10))
  plt.plot(observed)
  plt.plot(predicted, linestyle="dashed")
  #plt.title('Observation vr. prediction')
  plt.ylabel('Muon counts')
  plt.xlabel('Index')
  plt.legend(['Observed', 'Predicted'], loc='upper right')
  plt.show()

#Carga y organización de los datos
data = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/Datos_CR_Full.csv")
weather1 = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined1.csv")
weather2 = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined2.csv")
weather3 = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined3.csv")
weather4 = pd.read_csv("https://media.githubusercontent.com/media/CoPeMar/TFM/refs/heads/main/combined4.csv")
weather_full = pd.concat([weather1, weather2, weather3, weather4], ignore_index=True)
#Eliminamos el mes de febrero, para el que no tenemos datos
feb = np.linspace(1066464,1178519,num=1178520-1066464) 
weather_full.drop(weather_full.index[feb],inplace=True)
weather_full["time"] = weather_full["fecha"] + " " + weather_full["hora"]
dates = [dateutil.parser.parse(s) for s in data._time_]
weather_dates = [dateutil.parser.parse(s) for s in weather_full.time]
weather_full.index = weather_dates
data.index = dates
data.drop("_time_",axis=1,inplace=True)
weather_full.drop(["fecha","hora","time"],axis=1,inplace=True)
data_hour = data.resample("60min").mean()
xd = md.DateFormatter("%Y-%m-%d %H:%M:%S")
data_hour.dropna(inplace=True,axis=0,subset=["top","bottom","c8"])

if quiet_days_only:
    temp = pd.read_csv("C:/TFM_Data/Q_Days.txt")
    Q_Days = pd.DataFrame()
    Q_Days_2 = pd.DataFrame()
    Q_data_hour = pd.DataFrame()
    Q_weather_full = pd.DataFrame()
    for i in temp.columns[2:]:
        for j in temp.index:
            Q_Days.loc[j,i] = f"{temp.iloc[j,0]}-{temp.iloc[j,1]}-{temp.loc[j,i]}"
            
    for i in Q_Days.columns:
        for j in Q_Days.index:
            Q_Days_2.loc[j,i] = datetime.strptime(Q_Days.loc[j,i], "%Y-%m-%d")
    
    for i in range(len(Q_Days)):
        for j in range(len(Q_Days.columns)):
            Q_weather_full = pd.concat([Q_weather_full,weather_full[weather_full.index.normalize() == pd.Timestamp(Q_Days_2.iloc[i,j])]])
            Q_data_hour = pd.concat([Q_data_hour,data_hour[data_hour.index.normalize() == pd.Timestamp(Q_Days_2.iloc[i,j])]])
            
    weather_full = Q_weather_full
    data_hour = Q_data_hour

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
    event_rects.append(mpatches.Rectangle((md.date2num(datetime.strptime(event_dates[i], "%Y-%m-%d %H:%M:%S")), 0.9), md.date2num(datetime.strptime(event_dates[i+1], "%Y-%m-%d %H:%M:%S")) - md.date2num(datetime.strptime(event_dates[i], "%Y-%m-%d %H:%M:%S")), 0.3, color=color, alpha=0.3, label=label))

#Realizamos la media de los datos a lo largo de toda la altura para poder hacer la comparación
#weather_avg = weather_full.groupby(weather_full.index).mean()

#Agrupamos los datos en un solo dataframe
#comparison_df = pd.concat([data_hour, weather_avg], axis=1)
#comparison_df.drop(["ch01_LND", "ch02_LND", "ch03_LND", "theta(<6)", "theta(>6)", "atmpres_Pa","rel_humidity","temp_c","heightAboveSea","ch01_BP28","ch02_BP28","ch03_BP28","N-S","W-E"], axis=1, inplace=True)
#comparison_df.dropna(inplace=True)
#sns.heatmap(comparison_df.corr(numeric_only = True), annot=False, linewidths = 0.75, linecolor = "black")

#Creamos nuevos dataframes con los datos de presión, temperatura y humedad según su altura, para poder hacer la comparación con los datos de conteo
pres_df = pd.DataFrame(index=data_hour.index,columns=np.insert(weather_full["heightAboveSea"].unique(),5,2364))
t_df = pd.DataFrame(index=data_hour.index,columns=np.insert(weather_full["heightAboveSea"].unique(),5,2364))
r_df = pd.DataFrame(index=data_hour.index,columns=np.insert(weather_full["heightAboveSea"].unique(),5,2364))

for height in pres_df.columns:
    pres_df[height] = weather_full[weather_full["heightAboveSea"] == height]["pres"]
    t_df[height] = weather_full[weather_full["heightAboveSea"] == height]["t"]
    r_df[height] = weather_full[weather_full["heightAboveSea"] == height]["r"]
    
for i in range(len(pres_df.index)):
    pres_df.iloc[i,5] = np.interp(x=pres_df.columns[5],xp = [2000,2500],fp = pres_df.iloc[i,[4,6]])
    t_df.iloc[i,5] = np.interp(x=t_df.columns[5],xp = [2000,2500],fp = t_df.iloc[i,[4,6]])
    r_df.iloc[i,5] = np.interp(x=r_df.columns[5],xp = [2000,2500],fp = r_df.iloc[i,[4,6]])
    
pres_df.drop(pres_df.columns[0:5],axis=1,inplace=True)
t_df.drop(t_df.columns[0:5],axis=1,inplace=True)
r_df.drop(r_df.columns[0:5],axis=1,inplace=True)
   
#Juntamos presión y temperatura, que son los más interesantes, y hacemos PCA tras reescalar datos
scaler = StandardScaler()
pca = PCA(n_components=30)
if lowest_pressure_only:
    combined_df = pd.concat([pres_df.iloc[:,0], t_df], axis=1)
else:
    combined_df = pd.concat([pres_df, t_df], axis=1)
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
    loadings.loc[-1] = ["P" if i < 1 else "T" for i in range(len(loadings.T))]
else:
    loadings.loc[-1] = ["P" if i < 151 else "T" for i in range(len(loadings.T))]

#Escogemos los componentes principales a usar según su varianza explicada y su relación
#con los datos de conteo. Para esto, se ha establecido un límite de varianza explicada 
#del 1% y un límite de R^2 del 0.02 para considerar la relación entre la componente 
#principal y el conteo de partículas relevante. Se han guardado los índices de las 
#componentes principales relevantes para cada tipo de conteo (top, bottom y c8) para 
#poder hacer la comparación posteriormente.
index_top = []
suma = 0
for i in range(len(pca.explained_variance_ratio_)):
    if pca.explained_variance_ratio_[i] < explained_variance_threshold: #Límite de varianza explicada para considerar la componente principal relevante
        break
    #fig = plt.figure(i+1)
    #ax = plt.gca()
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean())
    r_squared = reg.score(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean())
    #print(r_squared)
    #print(reg.coef_)
    #plt.plot(pca_result[:,i],data_hour.top/data_hour["top"].mean(),".")
    #plt.plot(pca_result[:,i],reg.coef_*pca_result[:,i]+1)
    if r_squared > r_squared_threshold: #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
        suma += reg.coef_*pca_result[:,i]
        index_top.append(i)

r_squared_top = []
score_top = []
for i in range(len(pca.explained_variance_ratio_)):
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean())
    r_squared_top.append(reg.score(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean()))
    score_top.append(reg.coef_[0])
    
scoef_top = np.vstack((r_squared_top,score_top,pca.explained_variance_ratio_)).T
scoef_top = pd.DataFrame(scoef_top, columns=["R_squared","Score","Explained_Variance_Ratio"])
    
top_new = data_hour["top"]-(data_hour["top"].mean()*suma)       
fig = plt.figure(10)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación top original vs top corregido")
if quiet_days_only:
    sns.scatterplot(x=pres_df.index, y=data_hour["top"]/data_hour["top"].mean(), color='r',s=5)
    sns.scatterplot(x=pres_df.index, y=top_new/top_new.mean(), color='b',s=5)
else:
    plt.plot(pres_df.index, data_hour["top"]/data_hour["top"].mean(), color='r')
    plt.plot(pres_df.index, top_new/top_new.mean(), color='b')
plt.hlines(1, pres_df.index[0], pres_df.index[-1], colors="k", linestyles="dashed")
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

index_bottom = []
suma = 0
for i in range(len(pca.explained_variance_ratio_)):
    if pca.explained_variance_ratio_[i] < explained_variance_threshold:
        break
    #fig = plt.figure(i+1)
    #ax = plt.gca()
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean())
    r_squared = reg.score(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean())
    #print(r_squared)
    #print(reg.coef_)
    #plt.plot(pca_result[:,i],data_hour.top/data_hour["top"].mean(),".")
    #plt.plot(pca_result[:,i],reg.coef_*pca_result[:,i]+1)
    if r_squared > r_squared_threshold:
        suma += reg.coef_*pca_result[:,i]
        index_bottom.append(i)
    
r_squared_bottom = []
score_bottom = []
for i in range(len(pca.explained_variance_ratio_)):
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean())
    r_squared_bottom.append(reg.score(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean()))
    score_bottom.append(reg.coef_[0])
    
scoef_bottom = np.vstack((r_squared_bottom,score_bottom,pca.explained_variance_ratio_)).T
scoef_bottom = pd.DataFrame(scoef_bottom, columns=["R_squared","Score","Explained_Variance_Ratio"])

bottom_new = data_hour["bottom"]-(data_hour["bottom"].mean()*suma)       
fig = plt.figure(11)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación bottom original vs bottom corregido")
if quiet_days_only:
    sns.scatterplot(x=pres_df.index, y=data_hour["bottom"]/data_hour["bottom"].mean(), color='r',s=5)
    sns.scatterplot(x=pres_df.index, y=bottom_new/bottom_new.mean(), color='b',s=5)
else:
    plt.plot(pres_df.index,data_hour["bottom"]/data_hour["bottom"].mean(),"r")
    plt.plot(pres_df.index,bottom_new/bottom_new.mean(),"b")
plt.hlines(1, pres_df.index[0], pres_df.index[-1], colors="k", linestyles="dashed")
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

index_coin8 = []
suma = 0
for i in range(len(pca.explained_variance_ratio_)):
    if pca.explained_variance_ratio_[i] < explained_variance_threshold: #Límite de varianza explicada para considerar la componente principal relevante
        break
    #fig = plt.figure(i+1)
    #ax = plt.gca()
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.c8/data_hour["c8"].mean())
    r_squared = reg.score(pca_result[:,i].reshape(-1,1),data_hour.c8/data_hour["c8"].mean())
    #print(r_squared)
    #print(reg.coef_)
    #plt.plot(pca_result[:,i],data_hour.top/data_hour["top"].mean(),".")
    #plt.plot(pca_result[:,i],reg.coef_*pca_result[:,i]+1)
    if r_squared > r_squared_threshold: #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
        suma += reg.coef_*pca_result[:,i]
        index_coin8.append(i)

r_squared_coin8 = []
score_coin8 = []
for i in range(len(pca.explained_variance_ratio_)):
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.c8/data_hour["c8"].mean())
    r_squared_coin8.append(reg.score(pca_result[:,i].reshape(-1,1),data_hour.c8/data_hour["c8"].mean()))
    score_coin8.append(reg.coef_[0])
    
scoef_coin8 = np.vstack((r_squared_coin8,score_coin8,pca.explained_variance_ratio_)).T
scoef_coin8 = pd.DataFrame(scoef_coin8, columns=["R_squared","Score","Explained_Variance_Ratio"])
    
coin8_new = data_hour["c8"]-(data_hour["c8"].mean()*suma)       
fig = plt.figure(12)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación c8 original vs c8 corregido")
if quiet_days_only:
    sns.scatterplot(x=pres_df.index, y=data_hour["c8"]/data_hour["c8"].mean(), color='r', s=5)
    sns.scatterplot(x=pres_df.index, y=coin8_new/coin8_new.mean(), color='b', s=5)
else:
    plt.plot(pres_df.index,data_hour["c8"]/data_hour["c8"].mean(),"r")
    plt.plot(pres_df.index,coin8_new/coin8_new.mean(),"b")
plt.hlines(1, pres_df.index[0], pres_df.index[-1], colors="k", linestyles="dashed")
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

for i in range(len(index_top)):
    fig = plt.figure(20+i)
    ax = plt.gca()
    sns.scatterplot(x = loadings.iloc[index_top[i]].index, y = loadings.iloc[index_top[i]],hue=loadings.loc[-1],palette={"P":"b","T":"r"})
    ax.set_title(f"Loadings PC{index_top[i]+1} - top")
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Loading")

for i in range(len(index_bottom)):
    fig = plt.figure(40+i)
    ax = plt.gca()
    sns.scatterplot(x = loadings.iloc[index_bottom[i]].index, y = loadings.iloc[index_bottom[i]],hue=loadings.loc[-1],palette={"P":"b","T":"r"})
    ax = plt.gca()
    ax.set_title(f"Loadings PC{index_bottom[i]+1} - bottom")
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Loading")
    
for i in range(len(index_coin8)):
    fig = plt.figure(60+i)
    ax = plt.gca()
    sns.scatterplot(x = loadings.iloc[index_coin8[i]].index, y = loadings.iloc[index_coin8[i]],hue=loadings.loc[-1],palette={"P":"b","T":"r"})
    ax = plt.gca()
    ax.set_title(f"Loadings PC{index_coin8[i]+1} - coin8")
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Loading")
    
if lowest_pressure_only:
    pres_length = 1
    p_ori = "bo"
    p_corr = "go"
else:
    pres_length = 151
    p_ori = "b"
    p_corr = "b--"
test_cov = pd.concat([combined_df,data_hour[["top","bottom","c8"]],top_new,bottom_new,coin8_new],axis=1)
corr = test_cov.corr(numeric_only=True)
corr.drop(test_cov.columns[0:combined_df.shape[1]],axis=1,inplace=True)
corr.drop(test_cov.columns[combined_df.shape[1]+3:],axis=0,inplace=True)
corr.columns = ["top_original","bottom_original","c8_original","top_corregido","bottom_corregido","c8_corregido"]
fig = plt.figure(100)
ax = plt.gca()
plt.plot(corr.index[:pres_length],corr.top_original[:pres_length],p_ori,label="top original, presión")
plt.plot(corr.index[pres_length:],corr.top_original[pres_length:],"r",label="top original, temperatura")
plt.plot(corr.index[:pres_length],corr.top_corregido[:pres_length],p_corr,label="top corregido, presión")
plt.plot(corr.index[pres_length:],corr.top_corregido[pres_length:],"r--",label="top corregido, temperatura")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Correlation with top counts")
ax.set_title(f"Correlation top. R^2 = {r_squared_threshold}")
ax.legend()
fig = plt.figure(101)
ax = plt.gca()
plt.plot(corr.index[:pres_length],corr.bottom_original[:pres_length],p_ori,label="bottom original, presión")
plt.plot(corr.index[pres_length:],corr.bottom_original[pres_length:],"r",label="bottom original, temperatura")
plt.plot(corr.index[:pres_length],corr.bottom_corregido[:pres_length],p_corr,label="bottom corregido, presión")
plt.plot(corr.index[pres_length:],corr.bottom_corregido[pres_length:],"r--",label="bottom corregido, temperatura")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Correlation with bottom counts")
ax.set_title(f"Correlation bottom. R^2 = {r_squared_threshold}")
ax.legend()
fig = plt.figure(102)
ax = plt.gca()
plt.plot(corr.index[:pres_length],corr.c8_original[:pres_length],p_ori,label="c8 original, presión")
plt.plot(corr.index[pres_length:],corr.c8_original[pres_length:],"r",label="c8 original, temperatura")
plt.plot(corr.index[:pres_length],corr.c8_corregido[:pres_length],p_corr,label="c8 corregido, presión")
plt.plot(corr.index[pres_length:],corr.c8_corregido[pres_length:],"r--",label="c8 corregido, temperatura")
ax.set_xlabel("Height (m)")
ax.set_ylabel("Correlation with c8 counts")
ax.set_title(f"Correlation c8. R^2 = {r_squared_threshold}")
ax.legend()

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
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
    fig = plt.figure(200)
    ax = plt.gca()
    plot_loss(history)
    
    model_predictions = model.predict(X).flatten()
    model_predictions = scalerY.inverse_transform(model_predictions.reshape(-1,1)).flatten()
    rmse = root_mean_squared_error(data_hour["top"], model_predictions)
    pearson_corr, _ = pearsonr(data_hour["top"], model_predictions)
    
    plot_df = pd.DataFrame({"Observed": data_hour["top"], "Predicted": model_predictions})

    plot_predict(plot_df["Observed"], plot_df["Predicted"])
    print(f"RMSE: {rmse}")
    print(f"Pearson correlation: {pearson_corr}")
    
    #Predicción inversa?
    model_2 = Sequential()
    model_2.add(keras.Input(shape=(1,)))
    model_2.add(Dense(64, activation=activation))
    model_2.add(Dense(64, activation=activation))
    model_2.add(Dropout(dropout))
    model_2.add(Dense(len(index_top), activation='linear'))
    model_2.compile(optimizer=optimizer, loss='mean_squared_error')
    
    history_2 = model_2.fit(y_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(y_test, x_test), verbose=0)
    fig = plt.figure(201)
    ax = plt.gca()
    plot_loss(history_2)
    
    model_predictions_2 = model_2.predict(y).reshape(-1, len(index_top))
    rmse_2 = root_mean_squared_error(X, model_predictions_2)
    pearson_corr_2, _ = pearsonr(X.flatten(), model_predictions_2.flatten())
    
    plot_df_2 = pd.DataFrame(model_predictions_2, columns=[f"PC{index+1}" for index in index_top])
    plot_predict(X.flatten(), model_predictions_2.flatten())
    print(f"RMSE: {rmse_2}")
    print(f"Pearson correlation: {pearson_corr_2}")
    
    #Que mala :/