# -*- coding: utf-8 -*-
from SI_model_Baseline_model import SI_model_Baseline_model
from SI_model_base import SI_model_base
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR
class SI_model(SI_model_base):
    """
        SI defines the model for a SI .

         Fields:
         - n_k                : number of neighbors
         - if_model_fitted    : Internal use, check if the model has been fitted
         - modelsB            : initial estimator for each SI (ModelB)
         - estimators         : estimators that will be used in the cv
         - C                  : value of C
         - type               : model type
    """

    def __init__(self, n_k=1, C=0):
        """
        Params:
            - n_k    : number of neighbors
            - C      : value of C selected

        Call init parent's class method 
        """
        self.n_k=n_k
        self.if_model_fitted=False
        self.modelsB=[]
        self.estimators=[]
        self.C=float(C)
        self.type='LSVR'
        warnings.filterwarnings("ignore")
        super(self.__class__, self).__init__()

    def fit(self,X_train,C_train,SI_train):
        """   
        Fit method
        Returns the model fitted
        Params:
            - X_train            : X train data
            - C_train            : C train data
            - SI_train           : SI train data
        """
        model=LinearSVR(random_state=0,tol=0.0000000000001, fit_intercept=False, C=self.C, dual=False,loss='squared_epsilon_insensitive' )
        return SI_model_Baseline_model.fit(self,X_train,C_train,SI_train, model)
        
    def predict(self,X_test, C_test, SI_test): 
        """
        Predict method
        Returns the model predicts
        Params:
            - X_test            : X test data
            - C_test            : C test data
            - SI_test           : SI test data
        """
        return SI_model_Baseline_model.predict(self,X_test, C_test, SI_test)