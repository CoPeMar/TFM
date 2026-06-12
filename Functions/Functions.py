import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Literal

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
def plot_corr(pres_flag,hum_flag,corr,r_squared_threshold,sensor:_TYPES):
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
    plt.title(f"Correlation {sensor}. R^2 = {r_squared_threshold}")
    plt.legend()