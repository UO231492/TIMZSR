# -*- coding: utf-8 -*-
from sklearn.base import clone
from SI_model_base import SI_model_base
import warnings
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
class SI_model_MPLC_model(SI_model_base):
    """
        SI defines the model for a SI .

         Fields:
         - n_k                    : number of neighbors
         - estimator              : estimator that will be used in the cv
         - if_model_fitted        : Internal use, check if the model has been fitted
         - if_modelBase_fitted    : Internal use, check if the model base has been fitted
         - modelsB                : models for the first learning
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
        
    def fit(self,X_train,C_train,SI_train,model):
        """   
        Fit method
        Params: 
            - X_train            : X train data
            - C_train            : C train data
            - SI_train           : SI train data
            - model              : estimator that will be used in the cv
        """
        W=np.zeros((SI_train.shape[1],X_train.shape[1])) 

        for i in range(0,SI_train.shape[1]):
            C_train.columns=range(0,C_train.shape[1])
            removeNaN=~np.isnan(C_train[i])
            X=X_train.loc[X_train.index[removeNaN]]
            C=C_train[i][removeNaN]
            m=clone(model, safe=False)
            if X.shape[0]!=0 and C.shape[0]!=0:
                modelB=super(self.__class__, self).fit(m,X,C) #fit of the first learning
                for j in range(0,len(modelB.coef_)):
                    W[i,j]=modelB.coef_[j] #coefs of each model
            else:
                m.if_model_fitted=False
                modelB=m
            
            self.modelsB.append(modelB)
        for j in range(0,len(modelB.coef_)):
            estimator=clone(model, safe=False)
            SI_train_trasp=SI_train.T #SI transposed 
            SI_train_trasp=pd.DataFrame(SI_train_trasp)
            SI_train_trasp[SI_train_trasp.shape[1]]=1
           
            self.estimators.append(super(self.__class__, self).fit(estimator,SI_train_trasp,  W[:,j])) #fit of the second learning
        self.if_modelBase_fitted=True
        

    def predict(self,X_test,C_test,SI_test): 
        """
        Predict method
        Returns the predictions
        Params:
            - X_test            : X test data
            - C_test            : C test data
            - SI_test           : SI test data
        """
        SI_test=np.array(SI_test).T #SI transposed 
        X_test=pd.DataFrame(X_test)
        
        WTest=np.zeros((SI_test.shape[0],X_test.shape[1])) 
        predicts=[]
        for j in range(0,X_test.shape[1]):
            for i in range(0,SI_test.shape[0]):
                SI_test_trasp=[SI_test[i]]
                SI_test_trasp=pd.DataFrame(SI_test_trasp)
                SI_test_trasp[SI_test_trasp.shape[1]]=1
                SI_test_i=SI_test_trasp
                c=self.estimators[j].predict(SI_test_i) #calculate second learning predictions
                #c=np.dot(SI_test_i,self.estimators[j].coef_) 
                WTest[i,j]=c #coefs of each model
        for m in range(0,SI_test.shape[0]):
            c=0
            removeNaN=~np.isnan(C_test)
            X=X_test.loc[X_test.index[removeNaN[C_test.columns[m]]]]
            ps=[]
            for j in range(0,X.shape[0]):
                
                c=np.dot(X.iloc[j],WTest[m]) #calculate first learning predictions
                ps.append(c)
            predicts.append(pd.Series(ps))
        return predicts