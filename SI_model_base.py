# -*- coding: utf-8 -*-
class SI_model_base:
    """
        SI_model_base: defines the SI model.

         Fields:
         - if_set_random_object    : Internal use, check if random object has been set
         - if_modelBase_fitted     : Internal use, check if the model base has been fitted
         - estimator               : Estimator that will be used in the cv
         - predicts                : Predicts of the model
    """
        
    def __init__(self):
                
        # Internal use, check if random object has been set
        self.if_set_random_object=False
        # Internal use, check if the model has been fitted
        self.if_modelBase_fitted=False


    def fit(self,estimator,X,y=None):
        """
        Fit method
        Returns the estimator fitted
        Params:
            - estimator    : estimator that will be used in the cv
            - X            : X data
            - y            : y data
        """
       
        self.estimator=estimator.fit(X, y) #Fit train dataset
        return self.estimator 
         
    def predict(self,estimator,new_SI): 
        """
        Predict method
        Returns the predicts
        Params:
            - new_SI         : SI data to test
            - estimator      : estimator used to predict
        """
        self.predicts=estimator.predict(new_SI)
        return self.predicts
    
         
def _NearestNeighborsPredict(model, new_SI):
        """
        Predict nearest neighbors
        Returns the kneighbors of the model
        Params:
            - model        : estimator used to predict the nearest neighbors
            - new_SI       : SI data to test
        """
        return model.kneighbors(new_SI)
        

class SI_modelGeneratorExcep(Exception):
    """
    Class to throw the exception
    """
    pass