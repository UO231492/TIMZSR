# -*- coding: utf-8 -*-
import pandas as pd
import numpy;
from math import ceil
import random
import math
from sklearn.base import clone
import copy
from scipy import stats
import numpy as np
class CvSI:
    """
        CvSI defines the cross_validation for a SI model.

         Fields:
         - n_folds                : number of folds
         - n_reps                 : number of reps
         - scoring                : metric used as scoring in the score calculation 
         - predicts               : C predicts for each fold and rep [n_reps,n_folds]
         - cv_estimators          : Estimators used for each folds and rep [n_reps,n_folds]
         - score                  : result of scoreCalculation function (predict C vs real C) [n_reps,n_folds]
         - scoreMS                : result of scoreCalculation function for the mean system [n_reps,n_folds]
         - X_train                : X train dataset for each fold and rep [n_reps,n_folds]
         - C_train                : C train dataset for each fold and rep [n_reps,n_folds]
         - SI_train               : SI train dataset for each fold and rep [n_reps,n_folds]
         - X_test                 : X test dataset for each fold and rep [n_reps,n_folds]
         - C_test                 : C test dataset for each fold and rep [n_reps,n_folds]
         - SI_test                : SI test dataset for each fold and rep [n_reps,n_folds]
         - X_index                : X index with the split fold [n_reps,n_folds]
         - C_index                : C index with the split fold [n_reps,n_folds]
         - SI_index               : SI index with the split fold [n_reps,n_folds]
         - if_set_random_object   : Internal use, check if random object has been set
         - if_model_fitted        : Internal use, check if the model has been fitted
         - SI_estimator           : estimator that will be used in the cv
         - X                      : X data
         - SI                     : SI data
         - C                      : C data
    """
  
        
    def __init__(self,n_folds, n_reps, scoring):
        """
        Params:
            - n_folds      : number of folds 1>n_folds<=max_folds(at least one instance for each fold)
            - n_reps       : number of reps 1>=n_reps
            - scoring      : score function
            
            On params errors in type or values a CvSIGeneratorExcep exception will be raised.
        """
         # Checking parameters
        BaseErrMsg='Unsatisfied constraint in CvSIGenerator.__init(...)__: ';
        # Checking types
        if type(n_folds)!=int:
            raise CvSIGeneratorExcep(BaseErrMsg+'Integer attribute has not integer value');
       
        # Checking values
        if n_folds<=1:
            raise CvSIGeneratorExcep(BaseErrMsg+'n_folds<=1');
 
        if n_reps<1:
            raise CvSIGeneratorExcep(BaseErrMsg+'n_reps<1');
            
        # Storing the parameters       
        self.n_folds=n_folds
        self.n_reps=n_reps
        self.scoring=scoring
        
        # Creating the fields
        self.predicts=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.cv_estimators=numpy.zeros((n_reps,n_folds), dtype=object)
        self.score=numpy.zeros((n_reps,n_folds),dtype=numpy.ndarray)   
        self.scoreMS=numpy.zeros((n_reps,n_folds),dtype=numpy.ndarray)   
        self.X_train=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.C_train=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.SI_train=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.X_test=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.C_test=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.SI_test=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.X_index=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.SI_index=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
        self.C_index=numpy.zeros((n_reps,n_folds), dtype=pd.core.frame.DataFrame)
       
        # Internal use, check if random object has been set
        self.if_set_random_object=False
        # Internal use, check if the model has been fitted
        self.if_model_fitted=False
        
    def splits(self,XC_index,SI_index,X,SI,C,i):
        """
        Apply the splits of X,SI and C using the index foe each fold
        Returns:
                - X_train      : X dataset that will be used in the train
                - X_test       : X dataset that will be used in the test
                - SI_train     : SI dataset that will be used in the train
                - SI_test      : SI dataset that will be used in the test
                - C_train      : C dataset that will be used in the train
                - C_test       : C dataset that will be used in the test
                - data_foldsX  : [id,train/test used] for X
                - data_foldsSI : [id,train/test used] for SI
                - data_foldsC  : [id,train/test used] for C
        Params:
                - XC_index     : index of X and C
                - SI_index     : index of SI
                - X            : X data
                - SI           : SI data
                - C            : C data
                - i            : fold
        """
        data_foldsX=(XC_index[1] == i).astype(int) #Select the ids using the train/test for the current fold. 1-> train 0->test
        data_foldsX=pd.concat([pd.Series(XC_index[0]),pd.Series(data_foldsX)],axis=1, ignore_index=True) #Storage the [id,train/test used] for X

        data_foldsSI=(SI_index[1] == i).astype(int) #Select the ids using the train/test for the current fold. 1-> train 0->test
        data_foldsSI=pd.concat([pd.Series(SI_index[0]),pd.Series(data_foldsSI)],axis=1, ignore_index=True) #Storage the [id,train/test used] for SI

            
        data_foldsC=numpy.zeros((C.shape[0],C.shape[1] )) #Creating a new dataframe for storing where is used each category (train/test)
        data_foldsC=numpy.c_[XC_index[0] ,data_foldsC].astype(int)  #Adding the example ids of X (and C)

         
        data_foldsC[(XC_index[1] == i),1:]=1 # Marked as 1 (train) the categories used in X
        data_foldsC[:,1:][:,SI_index[(SI_index[1]==i)][0]]=0 # Remove SI (test)
        
        #select and save the train/test for each data
        C_train=C.iloc[XC_index[(XC_index[1]!= i)][0]]
        C_train=C_train[SI_index[(SI_index[1]!= i)][0]]

        C_test=C.iloc[XC_index[(XC_index[1]== i)][0]]
        C_test=C_test[SI_index[(SI_index[1]== i)][0]]
        
        SI_train=SI.iloc[:,SI_index[(SI_index[1]!= i)][0]]
        SI_test=SI.iloc[:,SI_index[(SI_index[1]== i)][0]]
        
        X_test=X.iloc[XC_index[(XC_index[1] == i)][0],]
        X_train=X.iloc[XC_index[(XC_index[1] != i)][0],]
        
        return X_train,X_test,SI_train,SI_test,C_train,C_test,data_foldsX,data_foldsSI,data_foldsC

    
    def GSAlpha(self,X,SI,C,SI_estimator):
        """
        Apply the search for param alpha
        Returns the estimator selected
        Params:
                - X            : X data
                - SI           : SI data
                - C            : C data
                - SI_estimator : estimator that will be used in the cv
        """
        ids=list(range(0,X.shape[0])) #Create array of ids for X and C 
        rd=random.Random()
        rd.seed(3232)
        rd.shuffle(ids) #Shuffle de array using the random method param
        rd.seed(rd.randint(0,2**31) ) #next seed for next rep
        
        XC_index=_split_folds(ids,SI.T.shape[0]) #split X (and C) in folds [id,fold]
        
     
        ids=list(range(0,SI.T.shape[0])) #Create array of ids for SI
        rd.shuffle(ids) #Shuffle de array using the random method param
        rd.seed(rd.randint(0,2**31) ) #next seed for next rep
        SI_index=_split_folds(ids,SI.T.shape[0]) #split in folds [id,fold]
        
        #alphas proposed
        alpha=[0,math.exp(-3),math.exp(-2),math.exp(-1),math.exp(0),math.exp(1),math.exp(2),math.exp(3)]
        
        #cv in th GS
        score_temp=[]
        for fold in range(1,len(ids)+1):

            SI.columns=list(range(0,SI.T.shape[0]))
            C.columns=list(range(0,SI.T.shape[0]))
            X_train,X_test,SI_train,SI_test,C_train,C_test,data_foldsX,data_foldsSI,data_foldsC=self.splits(XC_index,SI_index,X,SI,C,fold)
                
            for i in alpha: #for each value of C
                SI_estimator_i=copy.deepcopy(SI_estimator)
                SI_estimator_i.alpha=i
                
                SI_estimator_i.fit(X_train,C_train,SI_train) #model fitted for a value of C
                results=SI_estimator_i.predict(X_test,C_test,SI_test) #predicts for a value of C
                scoref=self.scoreCalculationParamsSelection(C_test,results) #score for a value of C
                if  len(scoref)!=0:
                    score_temp.append([scoref,i]) #save the scores
            
        alphasFolds=[]
        for alphaFold in range(0,len(score_temp),len(alpha)):
            id_alpha=numpy.where(numpy.array(score_temp[alphaFold:alphaFold+len(alpha)])[:,0]==min(score_temp[alphaFold:alphaFold+len(alpha)])[0])[0][0] #min score for each fold
            alphasFolds.append(alpha[id_alpha]) #save the scores
        SI_estimator_alpha=copy.deepcopy(SI_estimator)
        if(stats.mode(alphasFolds)[1]==1): #if the mode appears only 1 time
            mean_alphas=[]
            for j in alphasFolds:
                mean_alphas.append(numpy.array(score_temp)[numpy.where(numpy.array(score_temp)[:,1]==j)[0]][:,0].mean()) #mean total scores
            id_alpha=numpy.where(numpy.array(mean_alphas)[:,0]==min(numpy.array(mean_alphas)[:,0]))[0][0] #min of the total means
            SI_estimator_alpha.alpha=alphasFolds[id_alpha]
        else:
            SI_estimator_alpha.alpha=stats.mode(alphasFolds)[0][0] #mode
        print("Alpha selected: "+ str(SI_estimator_alpha.alpha))
        return SI_estimator_alpha
    

    def GSC(self,X,SI,C,SI_estimator):
        """
        Apply the search for param C
        Returns the estimator selected
        Params:
                - X            : X data
                - SI           : SI data
                - C            : C data
                - SI_estimator : estimator that will be used in the cv
        """
        ids=list(range(0,X.shape[0])) #Create array of ids for X and C 
        rd=random.Random()
        rd.seed(3232)
        rd.shuffle(ids) #Shuffle de array using the random method param
        rd.seed(rd.randint(0,2**31) ) #next seed for next rep
        
        XC_index=_split_folds(ids,SI.T.shape[0]) #split X (and C) in folds [id,fold]
        
     
        ids=list(range(0,SI.T.shape[0])) #Create array of ids for SI
        rd.shuffle(ids) #Shuffle de array using the random method param
        rd.seed(rd.randint(0,2**31) ) #next seed for next rep
        SI_index=_split_folds(ids,SI.T.shape[0]) #split in folds [id,fold]
        Cs_param=[0.001,0.01,0.1,1,10,100,1000] #Cs proposed
      
        #cv in th GS
        score_temp=[]
        for fold in range(1,len(ids)+1):
            SI.columns=list(range(0,SI.T.shape[0]))
            C.columns=list(range(0,SI.T.shape[0]))
            X_train,X_test,SI_train,SI_test,C_train,C_test,data_foldsX,data_foldsSI,data_foldsC=self.splits(XC_index,SI_index,X,SI,C,fold)
    
            for i in Cs_param: #for each value of C
                SI_estimator_i=copy.deepcopy(SI_estimator)

                SI_estimator_i.C=i
                
                SI_estimator_i.fit(X_train,C_train,SI_train) #model fitted for a value of C
                results=SI_estimator_i.predict(X_test,C_test,SI_test) #predicts for a value of C
                scoref=self.scoreCalculationParamsSelection(C_test,results) #score for a value of C
                if  len(scoref)!=0:
                    score_temp.append([scoref,i]) #save the scores
            
        
        CsFolds=[]
        for CFold in range(0,len(score_temp),len(Cs_param)):
            id_C=numpy.where(numpy.array(score_temp[CFold:CFold+len(Cs_param)])[:,0]==min(score_temp[CFold:CFold+len(Cs_param)])[0])[0][0] #min score for each fold
            CsFolds.append(Cs_param[id_C])  #save the scores
        SI_estimator_C=copy.deepcopy(SI_estimator)
        if(stats.mode(CsFolds)[1]==1): #if the mode appears only 1 time
            mean_Cs=[]
            for j in CsFolds:
                mean_Cs.append(numpy.array(score_temp)[numpy.where(numpy.array(score_temp)[:,1]==j)[0]][:,0].mean()) #mean total scores
            id_C=numpy.where(numpy.array(mean_Cs)[:,0]==min(numpy.array(mean_Cs)[:,0]))[0][0] #min of the total means
            SI_estimator_C.C=CsFolds[id_C]
        else:
            SI_estimator_C.C=stats.mode(CsFolds)[0][0] #mode
        print("C selected: "+ str(SI_estimator_C.C))
        return SI_estimator_C
    
    def select_parameters(self,X,SI,C,SI_estimator):
        """
        Apply the search taking into account the estimador used
        Returns the estimator selected
        Params:
                - X            : X data
                - SI           : SI data
                - C            : C data  
                - SI_estimator : estimator that will be used in the cv
        """
        if SI_estimator.type=='Ridge':
            SI_estimator=self.GSAlpha(X,SI,C,SI_estimator)
            
        if SI_estimator.type=='LSVR':
            SI_estimator=self.GSC(X,SI,C,SI_estimator)
            
        return clone(SI_estimator,safe=False)
                       
    def fit(self,SI_estimator,X, SI, C,rd=None,n_k=1):
        """
        Fit method
        Returns the estimators used in the cv
        Params:
            - SI_estimator : estimator that will be used in the cv
            - X            : X data
            - SI           : SI data
            - C            : C data
            - rd                 : random object (Default value: None)
            - n_k      : number of neibors
        """
         # Checking parameters
        BaseErrMsg='Unsatisfied constraint in CvSIGenerator.__fit(...)__: ';
        self.if_model_fitted=True
        if type(X)!=pd.core.frame.DataFrame or type(SI)!=pd.core.frame.DataFrame or type(C)!=pd.core.frame.DataFrame:
            raise CvSIGeneratorExcep(BaseErrMsg+'DataFrame attributes are not a pandas.core.frame.DataFrame');

        if min([X.shape[0],SI.shape[1],C.shape[0],C.shape[1]])<self.n_folds:
            raise CvSIGeneratorExcep(BaseErrMsg+'n_folds<max_folds');
        
        # If you don't pass a value in random object, set default value
        if rd==None and self.if_set_random_object==False:
            rd=random.Random()
            rd.seed(3232)
            self.set_random_object(rd)
        # If you pass a value in random object, set random value
        if self.if_set_random_object==False:            
            self.set_random_object(rd)
            
        #Dataframe parameters
        self.X=X
        self.SI=SI
        self.C=C
        
        self.SI_estimator_param=SI_estimator
        for j in range(1,self.n_reps+1): #Loop over the reps
            
            ids=list(range(0,self.X.shape[0])) #Create array of ids for X and C 
            self.randomObject.shuffle(ids) #Shuffle de array using the random method param
            self.randomObject.seed(self.randomObject.randint(0,2**31) ) #next seed for next rep
            
            XC_index=_split_folds(ids,self.n_folds) #split X (and C) in folds [id,fold]
         
            ids=list(range(0,self.SI.T.shape[0])) #Create array of ids for SI
            self.randomObject.shuffle(ids) #Shuffle de array using the random method param
            self.randomObject.seed(self.randomObject.randint(0,2**31) ) #next seed for next rep
            SI_index=_split_folds(ids,self.n_folds) #split in folds [id,fold]
            
            for i in range(1,self.n_folds+1):  #Loop over the folds
                print("--Rep "+str(j)+"----Fold "+str(i)+"--")
                #split in folds using the index
                X_train,X_test,SI_train,SI_test,C_train,C_test,data_foldsX,data_foldsSI,data_foldsC=self.splits(XC_index,SI_index,self.X,self.SI,self.C,i)
                
                #Grid search to select the best value for the param
                self.SI_estimator=self.select_parameters(X_train,SI_train,C_train,self.SI_estimator_param)
                
                #Storing the data in the fields
                self.X_test[j-1][i-1]=X_test #Save X
                self.C_test[j-1][i-1]=C_test #Save C
                self.SI_test[j-1][i-1]=SI_test #Save SI
                self.X_train[j-1][i-1]=X_train #Save X
                self.C_train[j-1][i-1]=C_train #Save C
                self.SI_train[j-1][i-1]=SI_train #Save SI
                self.X_index[j-1][i-1]=data_foldsX
                self.C_index[j-1][i-1]=data_foldsC
                self.SI_index[j-1][i-1]=data_foldsSI
                self.cv_estimators[j-1][i-1]=self.SI_estimator.fit(X_train,C_train,SI_train)
                self.predicts[j-1][i-1]=self.SI_estimator.predict(X_test,C_test,SI_test)
  
        return self.cv_estimators


    def set_random_object(self,randomObject):
        """
        Set random object
        Params:
            - randomObject: Random object (seed)
        """

        self.randomObject=randomObject
        self.if_set_random_object=True
        
    def scoreCalculation(self):
        """
        Apply score calculation in the cv
        Returns the score calculation for all the reps and the score for the mean system
        """
        if self.if_model_fitted==False:
            raise CvSIGeneratorExcep('Model no fitted!')     
        
        for j in range(1,self.n_reps+1): #Loop over the reps
            for i in range(1,self.n_folds+1):  #Loop over the folds
                scoreAll=[]
                scoreAllMeanSystem=[]
                c=0
                fit_value=np.mean(np.mean(self.C_train[j-1][i-1]))
                for s in self.C_test[j-1][i-1].columns:
                    real=self.C_test[j-1][i-1]
                    predictedMeanSystem=_mean_System(fit_value,real)#Mean system
                    removeNaN=~numpy.isnan(real)
                    realS=real[s][real[s].index[removeNaN[s]]]   
                    predictedS=self.predicts[j-1][i-1][c]
                    predictedM=predictedMeanSystem[c]
                    if realS.shape[0]!=0 and predictedS.shape[0]!=0:
                        scoreAll.append(self.scoring(realS,predictedS, multioutput='raw_values'))
                        scoreAllMeanSystem.append(self.scoring(realS,predictedM, multioutput='raw_values'))
                    c=c+1
                #save scores
                self.score[j-1][i-1]=numpy.sum(scoreAll)
                self.scoreMS[j-1][i-1]=numpy.sum(scoreAllMeanSystem)
                
    def scoreCalculationParamsSelection(self,C_test, predicts):
        """
        Apply score calculation in the GS
        Returns the score calculation in the GS
        Params:
            - C_test   : Real y
            - predicts : Predicts by the model
        """
        scoreAll=[]
        c=0
        for s in C_test.columns: 
            removeNaN=~numpy.isnan(C_test)
            if len(predicts)!=0:
                realS=C_test[s][C_test[s].index[removeNaN[s]]]   
                predictedS=predicts[c]
                if realS.shape[0]!=0 and len(predicts)!=0:
                    scoreAll.append(self.scoring(realS,predictedS, multioutput='raw_values'))
            c=c+1
        if len(scoreAll)==0:
            return (scoreAll)
        return(scoreAll[0])
def _mean_System(fit_value, real):
    predicted=[]
    removeNaN=~np.isnan(real)
    for i in removeNaN.columns:
        C_testLen=real.index[removeNaN[i]].shape[0]
        predicted.append(np.repeat(fit_value,C_testLen))
    fit_value=predicted
    return fit_value
        
def _split_folds(ids,n_folds):
    """
    Generates de folds indices for the cross-validation 
    Returns a dataset with the id and the fold for each example/category
    Params:
        - n_folds  : number of folds
        - ids      : array of ids to split
    """
    split_rate=len(ids)/n_folds #Split rate to cut the data
    modulo=len(ids)%n_folds #Modulo of the split rate
    folds=[] #array to store the folds data
    
    for i in range(1,ceil(split_rate)+1): #Loop over the ceiling value of split_rate
        
        folds=folds+list(range(1,n_folds+1)) #Add the list of folds (0,1,2..n_folds) to the array 
    if modulo!=0: #If it is not an exact number
        
        for j in range(1,(n_folds-modulo)+1): # Loop over the unnecessary folds
            folds.pop() #Remove the last number of the array
            
    data_index=pd.concat([pd.Series(ids),pd.Series(folds).T],axis=1, ignore_index=True) #Concatenate the ids and the folds data

    return data_index


class CvSIGeneratorExcep(Exception):
    """
    Class to throw the exception
    """
    pass