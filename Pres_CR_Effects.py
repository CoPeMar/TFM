import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import seaborn as sns
import dateutil
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#Carga y organización de los datos
data = pd.read_csv("C:/TFM_Data/Datos_CR_Full.csv")
weather1 = pd.read_csv("C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20230501_20231023/3cols/combined.csv")
weather2 = pd.read_csv("C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20231024_20240408/3cols/combined.csv")
weather3 = pd.read_csv("C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20240409_20250430/3cols/combined.csv")
weather4 = pd.read_csv("C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20250501_20250630/3cols/combined.csv")
weather_full = pd.concat([weather1, weather2, weather3, weather4], ignore_index=True)
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
   
#Juntamos presión y temperatura, que son los más interesantes, y hacemos PCA
pca = PCA(n_components=302)
combined_df = pd.concat([pres_df, t_df], axis=1)
combined_df.dropna(inplace=True,axis=1)
pca_result = pca.fit_transform(combined_df)

suma = 0
for i in range(15):
    #fig = plt.figure(i+1)
    #ax = plt.gca()
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean())
    print(reg.score(pca_result[:,i].reshape(-1,1),data_hour.top/data_hour["top"].mean()))
    print(reg.coef_)
    #plt.plot(pca_result[:,i],data_hour.top/data_hour["top"].mean(),".")
    #plt.plot(pca_result[:,i],reg.coef_*pca_result[:,i]+1)
    suma += reg.coef_*pca_result[:,i]
 
top_new = data_hour["top"]-(data_hour["top"].mean()*suma)       
fig = plt.figure(10)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación top original vs top corregido")
plt.plot(pres_df.index,data_hour["top"]/data_hour["top"].mean(),"r")
plt.plot(pres_df.index,top_new/top_new.mean(),"b")
ax.legend(["top original", "top corregido"])

suma = 0
for i in range(15):
    #fig = plt.figure(i+1)
    #ax = plt.gca()
    reg = LinearRegression().fit(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean())
    print(reg.score(pca_result[:,i].reshape(-1,1),data_hour.bottom/data_hour["bottom"].mean()))
    print(reg.coef_)
    #plt.plot(pca_result[:,i],data_hour.top/data_hour["top"].mean(),".")
    #plt.plot(pca_result[:,i],reg.coef_*pca_result[:,i]+1)
    suma += reg.coef_*pca_result[:,i]
 
bottom_new = data_hour["bottom"]-(data_hour["bottom"].mean()*suma)       
fig = plt.figure(11)
ax = plt.gca()
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación bottom original vs bottom corregido")
plt.plot(pres_df.index,data_hour["bottom"]/data_hour["bottom"].mean(),"r")
plt.plot(pres_df.index,bottom_new/bottom_new.mean(),"b")
ax.legend(["bottom original", "bottom corregido"])