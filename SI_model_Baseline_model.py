# -*- coding: utf-8 -*-
from SI_model_base import SI_model_base
import warnings
import numpy as np
import pandas as pd
class SI_model_Baseline_model(SI_model_base):
    """
        SI defines the model for a SI .

         Fields:
         - n_k                    : number of neighbors
         - estimator              : estimator that will be used in the cv
         - if_model_fitted        : Internal use, check if the model has been fitted
         - if_modelBase_fitted    : Internal use, check if the model base has been fitted
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

    def fit(self,X_train,C_train,SI_train,model):
        """   
        Fit method
        Returns the estimator fitted
        Params: 
            - X_train            : X train data
            - C_train            : C train data
            - SI_train           : SI train data
            - model              : estimator that will be used in the cv
        """
        SI_train=SI_train.T
        X=[]
        C=[]
        for i in range(0, SI_train.shape[0]):
            tempSI=pd.DataFrame(SI_train.iloc[i]).T
            tempSI.index=range(0,tempSI.shape[0])
            tempSI=pd.DataFrame(np.repeat(np.array(tempSI), X_train.shape[0], axis=0)) #Repeat SI for each example
            X_train.index=tempSI.index
            X.append(pd.concat([X_train,tempSI], axis=1)) #Concatenate X and SI
            C.append(C_train.iloc[:,i])
        X=pd.concat(X)
        C=pd.concat(C)
        X.index=range(0,X.shape[0])
        C.index=range(0,C.shape[0])
        
        removeNaN=~np.isnan(C)
        X=X.loc[X.index[removeNaN]]
        C=C[removeNaN]
        if X.shape[0]!=0 and C.shape[0]!=0:
            self.estimator=super(self.__class__, self).fit(model,X,C) #fit the model
        else:
            model.if_model_fitted=False
            self.estimator=model
        
        self.if_modelBase_fitted=True

        return self.estimator
        
    def predict(self,X_test, C_test, SI_test): 
        """
        Predict method
        Returns the predictions
        Params:
            - X_test            : X test data
            - C_test            : C test data
            - SI_test           : SI test data
        """
        SI_test=SI_test.T
        X=[]
        C=[]
        predicts=[]
        for i in range(0, SI_test.shape[0]):
            tempSI=pd.DataFrame(SI_test.iloc[i]).T
            tempSI.index=range(0,tempSI.shape[0])
            tempSI=pd.DataFrame(np.repeat(np.array(tempSI), X_test.shape[0], axis=0)) #Repeat SI for each example
            X_test.index=tempSI.index
            X.append(pd.concat([X_test,tempSI], axis=1)) #Concatenate X and SI
            C.append(C_test.iloc[:,i])
        c=0
        for i in X:
            removeNaN=~np.isnan(C)
            i=i.iloc[i.index[removeNaN[c]]]
            if i.shape[0]!=0:
                predicts.append(self.estimator.predict(i)) #predict the model
            else:
                predicts.append([])
            c=c+1
        return predicts