import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.linear_model import LinearRegression
import statistics
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def Pressure_Correction(combined_df,data_hour):
#Para realizar cualquiera de las correcciones que aparecen en De Mendonça et al. (2016)
#es necesario realizar una corrección previa por presión. Así, también podemos comparar
#una corrección solo de presión con una corrección total para entender los efectos que
#eliminamos. En este caso, seguiremos el procedimiento detallado en el anexo del paper
#mencionado anteriormente.

#La ecuación de los efectos de presión sigue una distribución exponencial, establecida
#en De Mendonça et al. (2016) como Ip = I0*e**(0.01*b*(P-P0)). Ip es el conteo de muones
#esperado cuando la presión es P, considerando un conteo inicial de I0 cuando la presión
#es P0. b es el coeficiente barométrico (%/hPa) que nos interesa. Para calcularlo, hay
#que buscar el mes en que el efecto de la presión es mayor que cualquier otro factor.
#Debe ser un mes para evitar efectos moduladores de larga duración.

    corr = 0
    idx = 0
    coef = 0

    for i in range(len(combined_df.index)-720):
        LP = pd.DataFrame(combined_df.iloc[:,0]-combined_df.iloc[i:i+720,0].mean(),
                          index=combined_df.index)
        LI = pd.DataFrame(np.log(data_hour["top"]/data_hour["top"][i:i+720].mean())*100,
                       index=combined_df.index)
        reg = LinearRegression().fit(LP,LI)
        aux = reg.score(LP,LI)
        if np.abs(aux) > np.abs(corr):
            corr = aux
            idx = i
            coef = reg.coef_

#Desde principios de Abril a principios de Mayo 2024. Hay una correlación muy reducida
#entre ambos valores, lo cual resulta bastante obvio si tenemos en cuenta que la relación
#presión-cuentas se vuelve negativa a nivel de mar. Realizamos ahora la corrección

    IPC = data_hour.top*np.exp(-0.01*coef[0][0]*(combined_df.iloc[:,0]-
                                           combined_df.iloc[idx:idx+720,0].mean()))
    var = statistics.variance(data_hour.top - IPC)/statistics.variance(data_hour.top)
    
    return IPC,var

def Pressure_Correction_2(combined_df,data_hour):
#La corrección realizada utilizando el método de De Mendonça et al. (2016) es bastante mala, 
#debido probablemente a la selección de un mes en lugar de utilizar todo el conjunto de datos. 
#Por lo tanto, realizamos una corrección de presión como en Duperier

    scaler = StandardScaler(with_std=False)
    LP = pd.DataFrame(scaler.fit_transform(combined_df.iloc[:,0].values.reshape(-1,1)),
                      index=combined_df.index)
    
    modelo = sm.OLS((data_hour.top-data_hour["top"].mean())/(data_hour["top"].mean()),
                    sm.add_constant(LP)).fit()
    top_new = data_hour.top/(1+modelo.params[0]*LP.iloc[:,0])
    var = statistics.variance(data_hour.top - top_new)/statistics.variance(data_hour.top)
    
    return top_new,var