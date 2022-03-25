# -*- coding: utf-8 -*-
from SI_model_base import SI_model_base
from SI_model_SR_model import SI_model_SR_model
import warnings
from sklearn.linear_model import Ridge
class SI_model(SI_model_base):
    """
        SI defines the model for a SI .

         Fields:
         - n_k                : number of neighbors
         - if_model_fitted    : Internal use, check if the model has been fitted
         - modelsB            : initial estimator for each SI (ModelB)
         - estimators         : estimators that will be used in the cv
         - alpha              : value of alpha
         - type               : model type
    """
 
    def __init__(self, n_k=1, alpha=0, FindNearestNeighbors='Euclidean'):
        """
        Params:
            - n_k                  : number of neighbors
            - alpha                : value of alpha selected
            - FindNearestNeighbors : type od distance to use in the nearest neighbors
            
        Call init parent's class method 
        """
        self.n_k=n_k
        self.if_model_fitted=False
        self.modelsB=[]
        self.estimators=[]
        self._modelB=[]
        self.alpha=float(alpha)
        self.type='Ridge'
        self.FindNearestNeighbors=FindNearestNeighbors
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
        model=Ridge(fit_intercept=False,random_state=0,alpha=self.alpha)
        return SI_model_SR_model.fit(self,X_train,C_train,SI_train, model, FindNearestNeighbors=self.FindNearestNeighbors)
        
    def predict(self,X_test,C_test,SI_test):
        """
        Predict method
        Returns the model predicts
        Params:
            - X_test            : X test data
            - C_test            : C test data
            - SI_test           : SI test data
        """
        return SI_model_SR_model.predict(self,X_test, C_test, SI_test)