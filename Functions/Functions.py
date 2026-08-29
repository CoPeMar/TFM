import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from typing import Literal
from scipy.signal import welch, periodogram
import scipy.optimize
import matplotlib.dates as md
import statsmodels.api as sm

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
        reg = LinearRegression().fit(pca_data[:,i].reshape(-1,1),(muon_data-muon_data.mean())/muon_data.mean())
        r2 = reg.score(pca_data[:,i].reshape(-1,1),(muon_data-muon_data.mean())/muon_data.mean())
        r_squared.append(r2)
        score.append(reg.coef_[0])
        if r2 > r2_threshold: #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
            sum += reg.coef_*pca_data[:,i]
            index.append(i)
            
    scoef = np.vstack((r_squared,score,pca.explained_variance_ratio_)).T
    scoef = pd.DataFrame(scoef, columns=["R_squared","Score","Explained_Variance_Ratio"])
    muon_corr = muon_data-(muon_data.mean()*sum) 
    
    return index, muon_corr, scoef
  
def correction_2(pca,ev_threshold,r2_threshold,muon_data,pca_data):
    index, r_squared, score = [], [], []
    sum = 0
    for i in range(len(pca.explained_variance_ratio_)):
        if pca.explained_variance_ratio_[i] < ev_threshold: #Límite de varianza explicada para considerar la componente principal relevante
            r2_threshold = 1 #Evitar que guarde más componentes principales.
        reg = LinearRegression().fit(pca_data[:,i].reshape(-1,1),(muon_data-muon_data.mean())/muon_data.mean())
        r2 = reg.score(pca_data[:,i].reshape(-1,1),(muon_data-muon_data.mean())/muon_data.mean())
        r_squared.append(r2)
        score.append(reg.coef_[0])
        if r2 > r2_threshold: #Límite de R^2 para considerar la relación entre la componente principal y el conteo de partículas relevante
            sum += reg.coef_*pca_data[:,i]
            index.append(i)
            
    scoef = np.vstack((r_squared,score,pca.explained_variance_ratio_)).T
    scoef = pd.DataFrame(scoef, columns=["R_squared","Score","Explained_Variance_Ratio"])
    muon_corr = muon_data/(1+sum) 
    
    return index, muon_corr, scoef
  
def plot_loadings(index_top,index_bottom,index_coin8,loadings):
    for i in list(set(index_top+index_bottom+index_coin8)):
        fig = plt.figure(20+i)
        ax = plt.gca()
        if i in index_top and i in index_bottom and i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - top, bottom, coin8")#recuerda añadir bottom y coin8 si quieres que se vea en la gráfica
        elif i in index_top and i in index_bottom:
          ax.set_title(f"Loadings PC{i+1} - top and bottom") #recuerda añadir bottom si quieres que se vea en la gráfica
        elif i in index_top and i in index_coin8:
          ax.set_title(f"Loadings PC{i+1} - top and coin8") #recuerda añadir coin8 si quieres que se vea en la gráfica
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
  x_mean = np.mean(x)
  x = (x-x_mean)/x_mean
  y = (y-x_mean)/x_mean
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
  plt.xlabel("Periodo (Días)")
  plt.ylabel("PSD (Horas)")
  plt.title(f"Welch PSD, {types}")
  plt.grid(True)
  plt.legend()
  
  plt.figure()
  mask_top = (period_daysx >= 0.3) & (period_daysx <= 3)
  mask_corr = (period_daysy >= 0.3) & (period_daysy <= 3)
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("PSD (Horas)")
  plt.title(f"Welch PSD, {types}, 7h - 3d")
  plt.grid(True)
  y_lim = plt.ylim()
  plt.vlines(x=1,ymin=0,ymax=y_lim[1],colors="k",linestyles="dashed",label="Ciclos Diarios")
  plt.vlines(x=0.5,ymin=0,ymax=y_lim[1],colors="m",linestyles="dashed",label="Ciclos Semidiarios")
  plt.legend()
  #plt.savefig(f"../../Imagenes/Welch_Short_PCA005.png")
  
  plt.figure()
  plt.semilogx(period_daysx[mask_top],
               psdy[positivex][mask_corr]/psdx[positivey][mask_top],
               label="Función de transferencia")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("Función de transferencia")
  plt.title(f"Función de transferencia Welch, {types}, 7h - 3d")
  plt.grid(True)
  y_lim = plt.ylim()
  plt.vlines(x=1,ymin=0,ymax=y_lim[1],colors="k",linestyles="dashed",label="Ciclos Diarios")
  plt.vlines(x=0.5,ymin=0,ymax=y_lim[1],colors="m",linestyles="dashed",label="Ciclos Semidiarios")
  plt.legend()
  #plt.savefig(f"../../Imagenes/Welch_Short_TF_PCA005.png")
  
  plt.figure()
  mask_top = (period_daysx >= 3) & (period_daysx <= 60)
  mask_corr = (period_daysy >= 3) & (period_daysy <= 60)
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("PSD (Horas)")
  plt.title(f"Welch PSD, {types}, 3d - 60d")
  plt.grid(True)
  y_lim = plt.ylim()
  plt.vlines(x=10,ymin=0,ymax=y_lim[1],colors="k",linestyles="dashed",label="Ciclos Climáticos aproximados")
  plt.vlines(x=27,ymin=0,ymax=y_lim[1],colors="m",linestyles="dashed",label="Rotación solar media")
  plt.xlim(left=2.65)
  plt.legend()
  #plt.savefig(f"../../Imagenes/Welch_Mid_PCA005.png")
  
  plt.figure()
  plt.semilogx(period_daysx[mask_top],
               psdy[positivex][mask_corr]/psdx[positivey][mask_top],
               label="Función de transferencia")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("Función de transferencia")
  plt.title(f"Función de transferencia Welch, {types}, 3d - 60d")
  plt.grid(True)
  y_lim = plt.ylim()
  plt.vlines(x=10,ymin=0,ymax=y_lim[1],colors="k",linestyles="dashed",label="Ciclos Climáticos aproximados")
  plt.vlines(x=27,ymin=0,ymax=y_lim[1],colors="m",linestyles="dashed",label="Rotación solar media")
  plt.xlim(left=2.65)
  plt.legend()
  #plt.savefig(f"../../Imagenes/Welch_Mid_TF_PCA005.png")
  return period_daysx,psdx[positivex],period_daysy,psdy[positivey]

def periodogram_comparison(x,y,types=_METHODS): #Comparación de los psd antes y después de corregir.
  x_mean = np.mean(x)
  x = (x-x_mean)/x_mean
  y = (y-x_mean)/x_mean
  fs = 1.0 #Muestras/hora
  freqx,psdx = periodogram(x.values,
                     fs=fs)
  freqy,psdy = periodogram(y.values,
                     fs=fs)
  positivex = freqx > 0
  positivey = freqy > 0
  period_daysx = 1/freqx[positivex]/24
  period_daysy = 1/freqy[positivey]/24
  plt.figure()
  plt.semilogx(period_daysx,psdx[positivex],label="Datos originales")
  plt.semilogx(period_daysy,psdy[positivey],label="Datos corregidos")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("PSD (Horas)")
  plt.title(f"Periodogram PSD, {types}")
  plt.grid(True)
  plt.legend()
  
  plt.figure()
  mask_top = (period_daysx >= 60) & (period_daysx <= period_daysx[0])
  mask_corr = (period_daysy >= 60) & (period_daysy <= period_daysx[0])
  plt.semilogx(period_daysx[mask_top],psdx[positivex][mask_top],label="Datos originales")
  plt.semilogx(period_daysy[mask_corr],psdy[positivey][mask_corr],label="Datos corregidos")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("PSD (Horas)")
  plt.title(f"Periodogram PSD, {types}, 60d - {period_daysx[0]}d")
  plt.legend()
  plt.grid(True)
  #plt.savefig(f"../../Imagenes/Periodogram_Short_PCA005.png")
  
  plt.figure()
  plt.semilogx(period_daysx[mask_top],
               psdy[positivex][mask_corr]/psdx[positivey][mask_top],
               label="Función de transferencia")
  plt.xlabel("Periodo (Días)")
  plt.ylabel("Función de transferencia")
  plt.title(f"Función de transferencia Periodograma, {types}, 60d - {period_daysx[0]}d")
  plt.legend()
  plt.grid(True)
  #plt.savefig(f"../../Imagenes/Periodogram_Short_TF_PCA005.png")
  return period_daysx,psdx[positivex],period_daysy,psdy[positivey]

def fit_sin(tt,yy,index,ylabel,title):
  #Para realizar fits de ciertas gráficas
  tt = np.array(tt)
  yy = np.array(yy)
  ff = np.fft.fftfreq(len(tt),1)
  Fyy = abs(np.fft.fft(yy))
  guess_freq = abs(ff[np.argmax(Fyy[1:])+1]) #Excluimos frecuencia 0
  guess_amp = np.std(yy)*2.**0.5
  guess_offset = np.mean(yy)
  guess = np.array([guess_amp,2.*np.pi*guess_freq,0.,guess_offset])
  
  def sinfunc(t,A,w,p,c): return A*np.sin(w*t+p)+c
  popt,pcov = scipy.optimize.curve_fit(sinfunc,tt,yy,p0=guess,maxfev=5000)
  A,w,p,c = popt
  f = w/(2.*np.pi)
  fitfunc = lambda t: A*np.sin(w*t+p)+c
  df = pd.DataFrame({"func":yy,"fitfunc":sinfunc(tt,A,w,p,c)},index=index)
  yd = md.DateFormatter("%Y-%m-%d")
  fig = plt.figure()
  ax = plt.gca()
  plt.plot(df.index,df.func)
  plt.plot(df.index,df.fitfunc,"k--",linewidth=4)
  plt.subplots_adjust(bottom=0.2)
  plt.xticks(rotation=80)
  ax.xaxis.set_major_formatter(yd)
  ax.set(ylabel=ylabel, title=title)
  return {"amp": A,"omega": w,"phase":p,"offset":c,"freq":f,"period":1./f,"fitfunc":fitfunc,"maxcov":np.max(pcov),"rawres":(guess,popt,pcov)}

def fit_2sin(df):
  #Esta función sirve para comparar las fluctuaciones diarias de muones con las 
  #existentes en documentación anterior.
  df = pd.DataFrame(df)
  df.columns=["counts"]
  df["datetime"] = pd.to_datetime(df.index)
  df = df.sort_values("datetime").copy()
  
  df["date"] = df["datetime"].dt.date
  df["hour"] = df["datetime"].dt.hour
  
  omega = 2*np.pi/24 #Frecuencia angular
  
  results = []
  
  for date,day in df.groupby("date"):
    if len(day) != 24:
      continue
    
    if set(day["hour"]) != set(range(24)):
      continue
    
    day = day.sort_values("hour")
    t = day["hour"].to_numpy(dtype=float)
    y = day["counts"].to_numpy(dtype=float)
    
    #Regresión armónica
    X = np.column_stack([np.ones(24),np.cos(omega*t),
                         np.sin(omega*t),
                         np.cos(2*omega*t),
                         np.sin(2*omega*t)])
    model = sm.OLS(y,X).fit()
    A,C0,D0,C1,D1 = model.params
    B0 = np.sqrt(C0**2 + D0**2) #Amplitud de variación diurna
    phi0 = np.arctan2(-D0,C0) #Fase de variación diurna
    B1 = np.sqrt(C1**2 + D1**2) #Amplitud de variación semidiurna
    phi1 = np.arctan2(-D1,C1) #Fase de variación semidiurna
    
    tmax0 = (-phi0/omega)%24 #Tiempo en que se alcanza el máximo diurno
    tmax1 = (-phi1/omega)%24 #Tiempo en que se alcanza el máximo semidiurno
    
    if tmax0 < 12:
      B0 *= -1
    if tmax1 < 12:
      B1 *= -1
    
    results.append({"date": date, "A":A, "B0":B0, "B1":B1, 
                    "tmax0":tmax0, "tmax1":tmax1,
                    "r_squared":model.rsquared,
                    "D_squared":model.ssr})
  
  results = pd.DataFrame(results)
  results["B0_percent"] = 100*results.B0/results.A
  results["B1_percent"] = 100*results.B1/results.A
  return results