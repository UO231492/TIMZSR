# -*- coding: utf-8 -*-
from SI_model_base import SI_model_base
from sklearn.base import clone
import warnings
import numpy as np
import pandas as pd
from FindNearestNeighborsCityblock import NearestNeighbors as FindNearestNeighborsCityblock
from FindNearestNeighborsEuclidean import NearestNeighbors as FindNearestNeighborsEuclidean
class SI_model_SR_model(SI_model_base):
    """
        SI defines the model for a SI .

         Fields:
         - n_k                    : number of neighbors
         - estimator              : estimator that will be used in the cv
         - if_model_fitted        : Internal use, check if the model has been fitted
         - if_modelBase_fitted    : Internal use, check if the model base has been fitted
         - modelsB                : models for the first learning
         -_modelB                 : Internal use foe save the modelB
         - estimators             : models for the second learning
    """   
    def __init__(self, n_k=1):
        """
        Params:
            - n_k      : number of neighbors

        Call init parent's class method 
        """
        warnings.filterwarnings("ignore")
        super(self.__class__, self).__init__()
        self.n_k=n_k
        self.modelsB=[]
        self.estimators=[]
        self._modelB=[]
    def fit(self,X_train,C_train,SI_train, model, FindNearestNeighbors):
        """   
        Fit method (Nearest neighbor)
        Params:
            Dataset parameters: 
                - X_train            : X train data
                - C_train            : C train data
                - SI_train           : SI train data
        """
        self._modelB=[]

        W=np.zeros((SI_train.shape[1],X_train.shape[1])) 
        for i in range(0,SI_train.shape[1]):
            C_train.columns=range(0,C_train.shape[1])
            removeNaN=~np.isnan(C_train[i])
            X=X_train.loc[X_train.index[removeNaN]]
            C=C_train[i][removeNaN]
            if X.shape[0]!=0 and C.shape[0]!=0:
                modelC=clone(model, False)
                modelB=super(self.__class__, self).fit(modelC,X,C) #fit of the first learning
                modelB.if_model_fitted=True
                self._modelB.append(modelB)
                self.modelsB.append(modelB)
            else:
                modelB=clone(model, False)
                modelB.if_model_fitted=False
                self._modelB.append(modelB)
                self.modelsB.append(modelB)

                
        if FindNearestNeighbors=='Cityblock':
            
            estimator=FindNearestNeighborsCityblock(n_neighbors=SI_train.shape[1]) #Nearest neighbors search
        else:
            estimator=FindNearestNeighborsEuclidean(n_neighbors=SI_train.shape[1]) #Nearest neighbors search
        
        self.estimator=super(self.__class__, self).fit(estimator,SI_train.T, C_train.T) #fit of the second learning (nearest neighbors) 
        
        self.if_modelBase_fitted=True

        
    def predict(self,X_test,C_test,SI_test):
    
        """
        Predict method (predict nearest modelB)
        Params:
            Dataset parameters: 
                - X_test            : X test data
                - C_test            : C test data
                - SI_test           : SI test data
        """

        SI_test=np.array(SI_test).T
        X_test=pd.DataFrame(X_test)
        #X_test[X_test.shape[1]]=1
        predicts=[]
        for i in range(0,SI_test.shape[0]):
            dist, neigh_ind=super(self.__class__, self).predict(self.estimator,[SI_test[i]]) #calculate second learning predictions (nearest neighbors)
            WSI=[]
            removeNaN=~np.isnan(C_test)
            X=X_test.loc[X_test.index[removeNaN[C_test.columns[i]]]]
            if X.shape[0]!=0:
                if dist[0][0]==0:              
                    for j in range(0,len(self._modelB)):
                        nn=neigh_ind[0][dist[0]==0]==j
                        if len(np.where(nn==True)[0])!=0 and self._modelB[neigh_ind[0][j]].if_model_fitted:
                            p=self._modelB[neigh_ind[0][j]].predict(X)
                            WSI.append(p)
                    WSI=pd.DataFrame(WSI).T
                    predicts.append(np.mean(WSI, axis=1))
                else:
                    for j in range(0,len(self._modelB)):
                        if self._modelB[neigh_ind[0][j]].if_model_fitted:
                            p=self._modelB[neigh_ind[0][j]].predict(X) #calculate first learning predictions
                            p=p*dist[0][j] #weight by distances (inverse distance)
                            WSI.append(p)
                    WSI=pd.DataFrame(WSI).T
                    predicts.append(np.sum(WSI, axis=1)/np.sum(dist[0]))
            else:
                predicts.append([])
        return predicts
  