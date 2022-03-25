# -*- coding: utf-8 -*-
import random;
from CVSI import CvSI
from ReadDatasets import ReadDatasets
from SI_model_SRRidge import SI_model as SI_model_SRRidge
from SI_model_SRLSVR import SI_model as SI_model_SRLSVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import time

def launch_SR(n_folds, model, metric,X, SI, C, FindNearestNeighbors):  
    """
    Launch MPLC experiment
    Returns the CvSI object fitted and the execution time
    Params:
        - n_folds               : number of folds 1>n_folds<=max_folds(at least one instance for each fold)
        - model                : estimator that will be used in the cv
        - metric               : metric used in the score calculation
        - X                    : X data
        - SI                   : SI data
        - C                    : C data
        - FindNearestNeighbors : distance of the nearest neighbors learning
    """
    r3=random.Random();
    r3.seed(3232);
    
    n_reps=3
    start_time_total=time.time()
       
    if model=='Ridge':
        SI_estimator=SI_model_SRRidge(n_k=1, FindNearestNeighbors=FindNearestNeighbors)
    if model=='LSVR':
        SI_estimator=SI_model_SRLSVR(n_k=1, FindNearestNeighbors=FindNearestNeighbors)
    
    #Init CvSI
    CvSIGenerator=CvSI(n_folds,n_reps,scoring=metric)
    
    #Set random object
    CvSIGenerator.set_random_object(r3)
    
    #CvSI fitted
    CvSIGenerator.fit(SI_estimator,X,SI,C)
    
    #ScoreCalculation
    CvSIGenerator.scoreCalculation()
                
    end_time_total=time.time()-start_time_total

    return CvSIGenerator, end_time_total

def print_data(CvSIGenerator,end_time_total):
    """
    Calculate scores and times
    Params:
        - CvSIGenerator     : CvSI object fitted
        - end_time_total    : execution time
    """
    scoressaved=[]
    scoressavedMS=[]
    for s in CvSIGenerator.score.mean(axis=0):
        scoressaved.append(np.mean(s))
        
    for ms in CvSIGenerator.scoreMS.mean(axis=0):
        scoressavedMS.append(np.mean(ms)) 
    #print results
    print("Score:"+str((np.mean(scoressaved)/np.mean(scoressavedMS))*100))  #Total mean score
    print("Time:"+str(end_time_total))
    
#--------------------------------------------------------------------------------------

FindNearestNeighbors="Cityblock" #Euclidean
folds=3
model="LSVR" #Ridge
metricStr="mean_squared_error" #mean_absolute_error

if metricStr=="mean_squared_error":
    metric=mean_squared_error
else:
    metric=mean_absolute_error
       

for n_folds in range(folds,folds+1):
   
    #-----------Air pollution datasets------------
    X,SI,C=ReadDatasets.load_Pollutant('NO2_HI')
    #X,SI,C=ReadDatasets.load_Pollutant('PST_HI')
    #X,SI,C=ReadDatasets.load_Pollutant('NO_HI')
    #X,SI,C=ReadDatasets.load_Pollutant('SO2_HI')
    
    #---------Communities and crime dataset-------
    #X,SI,C=ReadDatasets.load_Communities()
    
    #------------------R datasets-----------------
    #X,SI,C=ReadDatasets.load_R_5_5()
    #X,SI,C=ReadDatasets.load_R_10_5()
    #X,SI,C=ReadDatasets.load_R_50_5()
    #X,SI,C=ReadDatasets.load_R_100_5()
    #X,SI,C=ReadDatasets.load_R_5_15()
    #X,SI,C=ReadDatasets.load_R_10_15()
    #X,SI,C=ReadDatasets.load_R_50_15()
    #X,SI,C=ReadDatasets.load_R_100_15()
    #X,SI,C=ReadDatasets.load_R_5_25()
    #X,SI,C=ReadDatasets.load_R_10_25()
    #X,SI,C=ReadDatasets.load_R_50_25()
    #X,SI,C=ReadDatasets.load_R_100_25()
    
    #------------------S datasets-----------------
    #X,SI,C=ReadDatasets.load_S_5_5()   
    #X,SI,C=ReadDatasets.load_S_10_5()  
    #X,SI,C=ReadDatasets.load_S_50_5()   
    #X,SI,C=ReadDatasets.load_S_100_5()
    #X,SI,C=ReadDatasets.load_S_5_15()   
    #X,SI,C=ReadDatasets.load_S_10_15()   
    #X,SI,C=ReadDatasets.load_S_50_15()    
    #X,SI,C=ReadDatasets.load_S_100_15()   
    #X,SI,C=ReadDatasets.load_S_5_25()
    #X,SI,C=ReadDatasets.load_S_10_25()
    #X,SI,C=ReadDatasets.load_S_50_25()   
    #X,SI,C=ReadDatasets.load_S_100_25()
    
    
    CvSIGenerator,end_time_total=launch_SR(n_folds, model, metric, X, SI, C, FindNearestNeighbors)
    
    print_data(CvSIGenerator,end_time_total)
    