import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Literal
from scipy.signal import welch, periodogram

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
  
def correction(pca,ev_threshold,r2_threshold,muon_data,pca_data):
    index, r_squared, score = [], [], []
    sum = 0
    for i in range(len(pca.explained_variance_ratio_)):
        if pca.explained_variance_ratio_[i] < ev_threshold: #Límite de varianza explicada para considerar la componente principal relevante
            r2_threshold = 1 #Evitar que guarde más componentes principales.
        reg = LinearRegression().fit(pca_data[:,i].reshape(-1,1),muon_data/muon_data.mean())
        r2 = reg.score(pca_data[:,i].reshape(-1,1),muon_data/muon_data.mean())
        r_squared.append(r2)
        score.append(reg.coef_[0])
        if r2 > r2_threshold: #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
            sum += reg.coef_*pca_data[:,i]
            index.append(i)
            
    scoef = np.vstack((r_squared,score,pca.explained_variance_ratio_)).T
    scoef = pd.DataFrame(scoef, columns=["R_squared","Score","Explained_Variance_Ratio"])
    muon_corr = muon_data-(muon_data.mean()*sum) 
    
    return index, muon_corr, scoef
  
def plot_loadings(index_top,index_bottom,index_coin8,loadings):
    for i in list(set(index_top+index_bottom+index_coin8)):
        fig = plt.figure(20+i)
        ax = plt.gca()
        if i in index_top and i in index_bottom and i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - top, bottom and coin8")
        elif i in index_top and i in index_bottom:
          ax.set_title(f"Loadings PC{i+1} - top and bottom")
        elif i in index_top and i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - top and coin8")
        elif i in index_bottom and i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - bottom and coin8")
        elif i in index_top:
          ax.set_title(f"Loadings PC{i+1} - top")
        elif i in index_bottom:
          ax.set_title(f"Loadings PC{i+1} - bottom")
        elif i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - coin8")
        sns.scatterplot(x = loadings.iloc[i].index, y = loadings.iloc[i],hue=loadings.loc[-1],palette={"P":"b","T":"r","R":"g"})
        ax.set_xlabel("Height (m)")
        ax.set_ylabel("Loading")
  
_TYPES = Literal["top", "bottom", "c8"] 
_METHODS = Literal["PCA", "EFF", "EFF_M", "MSS", "Duperier", "ATE", "GRD", "MMP", "ATE+GRD",
                   "ATE+MMP", "GRD+MMP", "ATE+GRD+MMP"]     
def plot_corr(pres_flag,hum_flag,corr,r_squared_threshold,sensor:_TYPES,Corr_method:_METHODS):
    if pres_flag:
        pres_length = 1
        p_ori = "bo"
        p_corr = "go"
    else:
        pres_length = 151
        p_ori = "b"
        p_corr = "b--"
        
    plt.plot(corr.index[:pres_length],corr[sensor + "_original"][:pres_length],p_ori,label=(sensor + " original, presión"))
    if hum_flag:
        plt.plot(corr.index[pres_length:pres_length*2],corr[sensor + "_original"][pres_length:pres_length*2],"r",label=(sensor + " original, temperatura"))
        plt.plot(corr.index[pres_length*2:],corr[sensor + "_original"][pres_length*2:],"g",label=(sensor + " original, humedad"))
    else:
        plt.plot(corr.index[pres_length:],corr[sensor + "_original"][pres_length:],"r",label=(sensor + " original, temperatura"))
    
    plt.plot(corr.index[:pres_length],corr[sensor + "_corregido"][:pres_length],p_corr,label=(sensor + " corregido, presión"))
    if hum_flag:
        plt.plot(corr.index[pres_length:pres_length*2],corr[sensor + "_corregido"][pres_length:pres_length*2],"r--",label=(sensor + " corregido, temperatura"))
        plt.plot(corr.index[pres_length*2:],corr[sensor + "_corregido"][pres_length*2:],"g--",label=(sensor + " corregido, humedad"))
    else:
        plt.plot(corr.index[pres_length:],corr[sensor + "_corregido"][pres_length:],"r--",label=(sensor + " corregido, temperatura"))
    plt.xlabel("Height (m)")
    plt.ylabel(f"Correlation with {sensor} counts")
    plt.title(f"Correlation {sensor}. R^2 = {r_squared_threshold}. {Corr_method}")
    plt.legend()
    
def welch_comparison(x,y,types=_METHODS): #Comparación de los psd antes y después de corregir.
  freqx,psdx = welch(x.values,
                     fs=1.0, #Datos por hora
                     nperseg=4096) #Muestras por segmento.
  freqy,psdy = welch(y.values,
                     fs=1.0, #Datos por hora
                     nperseg=4096) #Muestras por segmento
  positivex = freqx > 0
  positivey = freqy > 0
  period_daysx = 1/freqx[positivex]/24
  period_daysy = 1/freqy[positivey]/24
  plt.figure()
  plt.semilogx(period_daysx,psdx[positivex],label="Datos originales")
  plt.semilogx(period_daysy,psdy[positivey],label="Datos corregidos")
  plt.xlabel("Period (Days)")
  plt.ylabel("PSD")
  plt.title(f"Welch PSD, {types}")
  plt.grid(True)
  plt.legend()
  
  plt.figure()
  mask_top = (period_daysx >= 0.3) & (period_daysx <= 3)
  mask_corr = (period_daysy >= 0.3) & (period_daysy <= 3)
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Period (Days)")
  plt.ylabel("PSD")
  plt.title(f"Welch PSD, {types}, 7h - 3d")
  plt.legend()
  plt.grid(True)
  
  plt.figure()
  mask_top = (period_daysx >= 3) & (period_daysx <= 60)
  mask_corr = (period_daysy >= 3) & (period_daysy <= 60)
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Period (Days)")
  plt.ylabel("PSD")
  plt.title(f"Welch PSD, {types}, 3d - 60d")
  plt.legend()
  plt.grid(True)
  
  return period_daysx,psdx[positivex],period_daysy,psdy[positivey]

def periodogram_comparison(x,y,types=_METHODS): #Comparación de los psd antes y después de corregir.
  x = x-np.mean(x)
  y = y-np.mean(y)
  fs = 1.0 #Muestras/hora
  freqx,psdx = periodogram(x.values,
                     fs=fs)
  freqy,psdy = periodogram(y.values,
                     fs=1.0)
  positivex = freqx > 0
  positivey = freqy > 0
  period_daysx = 1/freqx[positivex]/24
  period_daysy = 1/freqy[positivey]/24
  plt.figure()
  plt.semilogx(period_daysx,psdx[positivex],label="Datos originales")
  plt.semilogx(period_daysy,psdy[positivey],label="Datos corregidos")
  plt.xlabel("Period (Days)")
  plt.ylabel("PSD")
  plt.title(f"Periodogram PSD, {types}")
  plt.grid(True)
  plt.legend()
  
  plt.figure()
  mask_top = (period_daysx >= 60) & (period_daysx <= period_daysx[0])
  mask_corr = (period_daysy >= 60) & (period_daysy <= period_daysx[0])
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Period (Days)")
  plt.ylabel("PSD")
  plt.title(f"Periodogram PSD, {types}, 60d - {period_daysx[0]}d")
  plt.legend()
  plt.grid(True)
  
  return period_daysx,psdx[positivex],period_daysy,psdy[positivey]
  