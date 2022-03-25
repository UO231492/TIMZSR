# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ReadDatasets():

    def load_Pollutant(pollutant): 
        print("--------------------- AIR POLLUTANT EXPERIMENT "+str(pollutant)+"---------------------")
        data = pd.read_csv(r"datasets/pollutants/data.csv", delimiter=',', header = 0, encoding = "UTF-8")
        SITotal=data[['idEstacion','Autopista_norte','Autopista_sur','Autopista_este','Autopista_oeste','Industria_norte','Industria_sur','Industria_este','Industria_oeste','Ciudad_norte','Ciudad_sur','Ciudad_este','Ciudad_oeste','Mar_Ria_norte','Mar_Ria_sur','Mar_Ria_este','Mar_Ria_oeste']]
        X=data[['Estación','Dia_Noche','VV_HI','DD_Norte', 'DD_Sur', 'DD_Este', 'DD_Oeste','TMP_HI','HR_HI','PRB_HI','RS_HI','LL_HI']]
        
        #preprocessing SI
        SI=SITotal[SITotal['idEstacion']==1].iloc[0].to_frame()
        for i in range(2,SITotal["idEstacion"].max()+1):
            SI=SI.join(SITotal[SITotal['idEstacion']==i].iloc[0].to_frame())
        SI=SI.drop(['idEstacion'], axis=0)
        
        #C creation
        C=np.zeros((len(X),len(SI.columns))) 
        C[:]=np.nan
        for indice_fila, fila in data.iterrows():
            C[int(fila['id']-1),int(fila['idEstacion']-1)]=int(fila[pollutant])
        
        X=pd.DataFrame(X)
         
        X[X.shape[1]]=1
       
        return X, pd.DataFrame(SI), pd.DataFrame(C)

            
    def load_Communities():
        
        print("--------------------- COMMUNITIES AND CRIME EXPERIMENT---------------------")
        data = pd.read_csv(r"datasets/communities/CommunitiesSIMatrix.csv", delimiter=',', header = 0, encoding = "UTF-8")
        data=np.array(data)
   
    
        X=data[16:,1:102]
        SI=data[:16,102:]
        C=data[16:,102:]
        
        scaler = MinMaxScaler()
        SI = scaler.fit_transform(SI)
      
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
     
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)

    def load_R_5_5():
        print("--------------------- R^5,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_10_5():
        print("--------------------- R^10,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_50_5():
        print("--------------------- R^50,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_100_5():
        print("--------------------- R^100,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_5_15():
        print("--------------------- R^5,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_10_15():
        print("--------------------- R^10,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_50_15():
        print("--------------------- R^50,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_100_15():
        print("--------------------- R^100,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_5_25():
        print("--------------------- R^5,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_10_25():
        print("--------------------- R^10,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_50_25():
        print("--------------------- R^50,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_R_100_25():
        print("--------------------- R^100,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/R/X_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/R/S_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/R/C_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_5_5():
        print("--------------------- S^5,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_5_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_10_5():
        print("--------------------- S^10,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_10_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_50_5():
        print("--------------------- S^50,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_50_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_100_5():
        print("--------------------- S^100,5 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_100_5.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_5_15():
        print("--------------------- S^5,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_5_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_10_15():
        print("--------------------- S^10,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_10_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_50_15():
        print("--------------------- S^50,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_50_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_100_15():
        print("--------------------- S^100,15 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_100_15.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_5_25():
        print("--------------------- S^5,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_5_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_10_25():
        print("--------------------- S^10,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_10_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_50_25():
        print("--------------------- S^50,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_50_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
    
    def load_S_100_25():
        print("--------------------- S^5100,25 EXPERIMENT---------------------")
        X = pd.read_csv(r"datasets/S/X_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        SI = pd.read_csv(r"datasets/S/S_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        C = pd.read_csv(r"datasets/S/C_100_25.csv", delimiter=',', header = None, encoding = "UTF-8")
        X=pd.DataFrame(X)
        X[X.shape[1]]=1
        return X, pd.DataFrame(SI), pd.DataFrame(C)
