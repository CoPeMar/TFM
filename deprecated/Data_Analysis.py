import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import seaborn as sns
import dateutil

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
weather_full.drop(["fecha","hora"],axis=1,inplace=True)
data_hour = data.resample("60min").mean()
xd = md.DateFormatter("%Y-%m-%d %H:%M:%S")
fig = plt.figure(1)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación BP28")

#plt.plot(data.dates['2024-10-17 00:00:00':'2024-10-18 00:00:00'],data.ch01_LND['2024-10-17 00:00:00':'2024-10-18 00:00:00'])

a=0
for i in ["r", "b", "k"]:
    plt.plot(data_hour.index, data_hour.iloc[:,a]/data_hour.iloc[:,a].mean(),i)
    a +=1

ax.legend(data.columns[0:3])

fig = plt.figure(2)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación LND")

for i in ["r", "b", "k"]:
    plt.plot(data_hour.index, data_hour.iloc[:,a]/data_hour.iloc[:,a].mean(),i)
    a +=1

ax.legend(data.columns[3:6])

fig = plt.figure(3)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Relative counts", title="Comparación Direcciones")

for i in ["r", "b"]:
    plt.plot(data_hour.index, data_hour.iloc[:,a]/data_hour.iloc[:,a].mean(),i)
    a +=1

ax.legend(data.columns[6:8])

fig = plt.figure(4)
sns.heatmap(data.corr(numeric_only = True), annot=False, linewidths = 0.75, linecolor = "black")

fig = plt.figure(5)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="counts", title="Incidencias")

for i in ["r", "b"]:
    plt.plot(data_hour.index, data_hour.iloc[:,a],i)
    a +=1
fig = plt.figure(6)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="counts", title="Theta")

for i in ["r", "b"]:
    plt.plot(data_hour.index, data_hour.iloc[:,a],i)
    a +=1

ax.legend(data.columns[10:12])

fig = plt.figure(5)
ax = plt.gca()
plt.plot(data_hour.index, data_hour.iloc[:,a],"k")
a +=1
ax.legend(["bottom", "c8", "top"])

fig = plt.figure(7)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Presión (hPa)", title="Presión a 1000 m")
plt.plot(weather_full[weather_full["heightAboveSea"] == 1000].index,weather_full[(weather_full["heightAboveSea"] == 1000)]["pres"],"r")

fig = plt.figure(8)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Humedad", title="Humedad a 1000 m")
plt.plot(weather_full[weather_full["heightAboveSea"] == 1000].index,weather_full[(weather_full["heightAboveSea"] == 1000)]["r"],"b")

fig = plt.figure(9)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=80)
ax = plt.gca()
ax.xaxis.set_major_formatter(xd)
ax.set(ylabel="Temperatura (ºC)", title="Temperatura a 1000 m")
plt.plot(weather_full[weather_full["heightAboveSea"] == 1000].index,weather_full[(weather_full["heightAboveSea"] == 1000)]["t"],"k")
