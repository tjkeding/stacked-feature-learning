#!/usr/bin/python3
# coding: utf-8

# =======================================================================================================================
# FEATURE LEARNING using STACKED GENERALIZATION and PERMUTATION IMPORTANCE ANALYSIS
# Written by Taylor J. Keding (tjkeding@gmail.com)
# Last Updated: 04.24.20
# =======================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS:
# -----------------------------------------------------------------------------------------------------------------------
# Python:
import sys
import string
import os
import re
import csv
import copy
import math
from time import sleep
from datetime import datetime
import multiprocessing as mp

# Joblib:
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

# Scikit-Learn:
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, \
	GradientBoostingClassifier,GradientBoostingRegressor,BaggingClassifier,BaggingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.svm import SVC,SVR
from sklearn.model_selection import ShuffleSplit,RandomizedSearchCV,StratifiedShuffleSplit,\
	cross_val_score,StratifiedKFold,KFold,cross_val_predict,LeaveOneOut
from sklearn.metrics import recall_score,precision_score,make_scorer,mean_squared_error,\
	accuracy_score,mean_absolute_error,average_precision_score,precision_recall_curve,roc_curve,\
	auc,f1_score,roc_auc_score,balanced_accuracy_score,make_scorer,r2_score
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.linear_model import SGDRegressor,SGDClassifier,ElasticNet
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,RBF,WhiteKernel
from sklearn.cluster import OPTICS
from sklearn.calibration import CalibratedClassifierCV

# Numpy:
import numpy as np

# Pandas:
import pandas as pd

# SciPy:
from scipy.stats import truncnorm,loguniform,uniform,spearmanr,pearsonr,norm,rankdata

# ------------ STOP WARNINGS ------------
import warnings
# Stop warnings related to lack of convergence 
# max_iter = 10000 already substantially > what is recommended
os.environ["PYTHONWARNINGS"] = "ignore"
# ---------------------------------------

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Start the timer
startTime = datetime.now()

# -----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS:
# -----------------------------------------------------------------------------------------------------------------------

# ============================================ READ, SAMPLE, & FORMAT DATA ==============================================

def check_args_read_data(args):
	print("")
	print("   CHECK ARGUMENTS & READ-IN DATA")
	print("   ---------------------------------------------------------")

	if(len(sys.argv)!= 9):
		print("Incorrect number of args! 8 Required:")
		printUsage()
		sys.exit()

	# Check args[1]: read-in CSV feature learning file
	try:
		df = pd.read_csv(args[1])
	except:
		print("Data file could not be found!")
		printUsage()
		sys.exit()

	# Check args[2]: get prefix for output files
	prefix = str(args[2])

	# Check args[3]: Check if numSims is in correct format
	try:
		numSims = int(args[3])
	except:
		print("<numSims> is incorrect (must be an int)")
		printUsage()
		sys.exit()

	# Check args[4]: Check if testProp is in the correct format
	try:
		testProp = float(args[4])
		# Check for the correct range (assumes no less than 1/50 and no more than 49/50)
		if testProp >= 0.98  or testProp <= 0.02:
			print("<testProp> is not in range (must be less than 0.98 and greater than 0.02)")
			printUsage()
			sys.exit()
	except:
		print("<testProp> is incorrect (must be a float, where 0.02 < <testProp> < 0.98)")
		printUsage()
		sys.exit()

	# Check args[5]: Check if kCVParam is in correct format
	try:
		kCVParam = int(args[5])
	except:
		print("<kCVParam> is incorrect (must be an int)")
		printUsage()
		sys.exit()

	# Check args[6]: Check if nSampling is correct
	try:
		nSampling = int(args[6])
		if nSampling <= 1:
			print("<nSampling> is incorrect (must be an int > 1)")
			printUsage()
			sys.exit()
	except:
		print("<nSampling> is incorrect (must be an int > 1)")
		printUsage()
		sys.exit()

	# Check args[7]: Check if numSimsPI is in correct format
	try:
		numSimsPI = int(args[7])
	except:
		print("<numSimsPI> is incorrect (must be an int)")
		printUsage()
		sys.exit()

	# Check args[8]: check if the number of cores is correct
	try:
		numCores = int(args[8])
		if(numCores > mp.cpu_count() or (numCores < 1 and numCores != -1)):
			print(str("Specifed too many (or too few) cores! 0 < Integer <= "+str(mp.cpu_count())))
			print(str("If you want to use all available, input '-1"))
			printUsage()
			sys.exit()
	except:
		print("Number of cores is not an integer!")
		printUsage()
		sys.exit()

	return {'df':df,'prefix':prefix,'numSims':numSims,'testProp':testProp,'kCVParam':kCVParam, \
		'nSampling':nSampling,'numSimsPI':numSimsPI,'numCores':numCores}

# --------------------

def vectorizeDataFrame(dataframe,labelCol,startOfFeats):

	x = dataframe[dataframe.columns[startOfFeats:len(dataframe.columns)]].values
	y = dataframe[dataframe.columns[labelCol]]

	return {'x':x,'y':y}

# --------------------

def getKFoldBatches(choice,labelCol,startOfFeats,allData,numBatches,testProp):
    
	outDict={}

	# For classification
	# Make sure there are balanced labels (as close as possible) for each set
	if(choice == 0):
		ss1 = StratifiedShuffleSplit(n_splits=numBatches,test_size=testProp,random_state=np.random.RandomState())
	else:
		ss1 = ShuffleSplit(n_splits=numBatches,test_size=testProp,random_state=np.random.RandomState())

	# Vectorize the dataset to be split
	vecData = vectorizeDataFrame(allData,labelCol,startOfFeats)

	# Keep track for batch names in the dictionary
	batch = 1
	splits = ss1.split(vecData['x'], vecData['y'])

	for train_index, test_index in splits:
	
		# Split data based on indices
		X_train, X_test = vecData['x'][train_index], vecData['x'][test_index]
		y_train, y_test = vecData['y'][train_index], vecData['y'][test_index]
		trainIDs = allData[allData.columns[0]].values[train_index]
		testIDs = allData[allData.columns[0]].values[test_index]
		y_train = list(y_train)
		y_test = list(y_test)

		# Get current training set as a dataframe
		trainDF = pd.DataFrame(X_train, columns=allData.columns[startOfFeats:len(allData.columns)])
		trainDF.insert(0, str(allData.columns[labelCol]), y_train)
		trainDF.insert(0, str(allData.columns[0]), trainIDs)

		# Get current test set to split
		testDF = pd.DataFrame(X_test, columns=allData.columns[startOfFeats:len(allData.columns)])
		testDF.insert(0, allData.columns[labelCol], y_test)
		testDF.insert(0, str(allData.columns[0]), testIDs)

		outDict[str(batch)]={'trainDF':trainDF,'testDF':testDF}
		batch = batch + 1

	return outDict

# ============================================== PROCESSING & STATS TOOLS ===============================================

def getScores(choice,true,pred,probs):
    	
	out = {}

	# Return all scores depending on what is being predicted
	if(choice == 0):
		out['balAcc'] = balanced_accuracy_score(true,pred)
		out['rawAcc'] = accuracy_score(true,pred)
		out['rocAUC'] = roc_auc_score(true,probs,average="weighted")
		out['f1'] = f1_score(true,pred,average="weighted",zero_division=0)
		out['precision'] = precision_score(true,pred,average="weighted",zero_division=0)
		out['recall'] = recall_score(true,pred,average="weighted",zero_division=0)
		out['avePrecis'] = average_precision_score(true,pred,average="weighted")
	else:
		out['MAE'] = mean_absolute_error(true,pred)
		out['MSE'] = mean_squared_error(true,pred)
		out['RMSE'] = mean_squared_error(true,pred,squared=False)
		out['r'], r_p = pearsonr(true,pred)
		out['r2'] = r2_score(true,pred)

	return out

# --------------------

def standardizeBatch(batch,labelCol,startOfFeats):

	trainDF = batch['trainDF']
	testDF = batch['testDF']

	for j in range(startOfFeats,len(trainDF.columns)):
		currMean = np.mean(trainDF[trainDF.columns[j]])
		currSD = np.std(trainDF[trainDF.columns[j]])

		trainDF[trainDF.columns[j]]=(trainDF[trainDF.columns[j]]-currMean)/currSD
		testDF[testDF.columns[j]]=(testDF[testDF.columns[j]]-currMean)/currSD

	return {'trainDF':trainDF,'testDF':testDF}

# --------------------

def calcPFromDist(score,side,dist):

	count = 0
	for i in dist:
		if side == "greater than":
			if i >= score:
				count += 1
		elif side == "less than":
			if i <= score:
				count += 1
	if count == 0:
		return str("< "+str(1/len(dist)))
	else:
		return count/len(dist)
	
# ==================================================== OUTPUT TOOLS =====================================================

def printUsage():
    	print("python3 stacked-feature-learning.py <dataFileName> <outputPrefix> <numSims> <testProp> <kCVParam> <nSampling> <numSimsPI> <numCores>")

# --------------------

def printSLCoefReport(choice,SL,labels,batchNum):
    	
	# Print super learner coefficients to command line
	# Return the same report as dataframe
	numLabels = len(set(labels))
	toReturn = {'BATCH':list([]),'MODEL':list([]),'COEF':list([])}

	print("")
	print("   SUPER LEARNER COEFFICIENTS:")
	arbCount = 0
	for i, key in enumerate(SL['models']):
		if choice == 0:
			for j in range(1,numLabels):
				if numLabels == 2:
					n = j - 1
				else:
					n = j
				print("   "+str(key)+"_"+str(j)+": "+str(SL['SL'].coef_[n][arbCount]))
				toReturn['BATCH'].append(batchNum)
				toReturn['MODEL'].append(str(str(key)+"_"+str(j)))
				toReturn['COEF'].append(SL['SL'].coef_[n][arbCount])
				arbCount+=1
		else:
			print("   "+str(key)+": "+str(SL['SL'].coef_[i]))
			toReturn['BATCH'].append(batchNum)
			toReturn['MODEL'].append(str(key))
			toReturn['COEF'].append(SL['SL'].coef_[i])

	return pd.DataFrame.from_dict(toReturn)

# --------------------

def printScoreReport(sub_scores,pVals,label,batchNum):
    
	# Print super learner and submodel scores to command line
	# Return the same report as dataframe
	scoresOut = {'BATCH':list([]),'MODEL':list([]),'STAT':list([]),'SCORE':list([]),'P':list([])}
	print("")
	if label == "test":
		print("   TEST SCORES:")
	else:
		print("   TRAIN SCORES")
	print("")
	print("   Super Learner")
	for key in sub_scores['SL']:
		print("   "+str(key)+": "+str(sub_scores['SL'][key])+", p-value: "+str(pVals['SL'][key]))
		scoresOut['BATCH'].append(batchNum)
		scoresOut['MODEL'].append("SL")
		scoresOut['STAT'].append(str(key))
		scoresOut['SCORE'].append(sub_scores['SL'][key])
		scoresOut['P'].append(pVals['SL'][key])
	print("")
	for key1 in sub_scores:
		if key1 != "SL":
			print("   "+str(key1))
			for key2 in sub_scores[key1]:
				print("   "+str(key2)+": "+str(sub_scores[key1][key2])+", p-value: "+str(pVals[key1][key2]))
				scoresOut['BATCH'].append(batchNum)
				scoresOut['MODEL'].append(str(key1))
				scoresOut['STAT'].append(str(key2))
				scoresOut['SCORE'].append(sub_scores[key1][key2])
				scoresOut['P'].append(pVals[key1][key2])
			print("")

	return pd.DataFrame.from_dict(scoresOut)

# --------------------

def saveFLReport(clusterLabels,featStats,batchNum):
    	
	# Return feature learning scores and p-vals as dataframe	
	toReturn = {'BATCH':list([]),'FEAT':list([]),'CLUST':list([]),'SCORE':list([]),'P':list([])}
	for i, key in enumerate(featStats):
		toReturn['BATCH'].append(batchNum)
		toReturn['FEAT'].append(str(key))
		toReturn['CLUST'].append(clusterLabels[i])
		toReturn['SCORE'].append(featStats[key][0])
		toReturn['P'].append(featStats[key][1])

	return pd.DataFrame.from_dict(toReturn)

# --------------------

def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill='█'):
    """
	Print iterations progress
	Adapted from work by Greenstick (https://stackoverflow.com/users/2206251/greenstick)
	and Chris Luengo (https://stackoverflow.com/users/7328782/cris-luengo)
	in response to https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# --------------------

def mergePoolOutput(choice,currStep,poolOutput):

	toReturn = {}

	# Merge parallel processing output for clustering
	if currStep == "clustering":
		for i, currDict in enumerate(poolOutput):	
			if i==0:
				toReturn['silCoef'] = [0.0]*len(poolOutput)
				toReturn['clustLabels'] = [0.0]*len(poolOutput)
			toReturn['silCoef'][i] = currDict['silCoef']
			toReturn['clustLabels'][i] = currDict['clustLabels']

	# Merge parallel processing output for model performance
	elif currStep == "modelBuilding":
		bestScore = sys.float_info.max
		if choice == 0:
			bestScore = -1*(sys.float_info.max)
		for mbDict in poolOutput:
			if (mbDict['score'] > bestScore and choice == 0) or \
				(mbDict['score'] < bestScore and choice == 1):
				bestScore = mbDict['score']
				toReturn['bestScore'] = bestScore
				toReturn['bestParams'] = mbDict['params']

	# Merge parallel processing output for model performance
	elif currStep == "nullScores":
		toReturn['test'] = {}
		toReturn['train'] = {}	
		for i in range(0,len(poolOutput)):
			for key in poolOutput[i]['test']:
				if i == 0:
					toReturn['test'][key] = {}
					toReturn['train'][key] = {}
				for key1 in poolOutput[i]['test'][key]:
					if i == 0:
						toReturn['test'][key][key1] = [0.0]*len(poolOutput)
						toReturn['train'][key][key1] = [0.0]*len(poolOutput)
					toReturn['test'][key][key1][i] = poolOutput[i]['test'][key][key1]
					toReturn['train'][key][key1][i] = poolOutput[i]['train'][key][key1]

	# Merge parallel processing output for feature importance
	elif currStep == "featImp":
		for i in range(0,len(poolOutput)):
			for key in poolOutput[i]:
				toReturn[key] = poolOutput[i][key]

	return toReturn
				
# --------------------

def saveDFtoFile(DF,filename):

	if os.path.exists(filename):
  		os.remove(filename)
	DF.to_csv(filename,index=False)

# =================================================== MODEL BUILDING ====================================================

def buildSuperLearner(choice,DFs,submodels,labelCol,startOfFeats,kCVParam,nSamp,numCores):

	# Standardize current batch (to mean/SD of current training set)
	normBatch = standardizeBatch(DFs,labelCol,startOfFeats)

	# trainDF used for hyperparameter tuning and training
	trainDF = normBatch['trainDF']

    # Hyperparameter Turning and model training
    # Returns a dictionary of optimized/trained models and hold-out predictions from trainDF
	# model names (from 'models') are the keys for the first dict
	print("")
	print("   BUILDING & OPTIMIZING MODELS FOR THE SUPER LEARNER")
	print("   ---------------------------------------------------------")
	paramSet = optimizeSubmodels(choice,submodels,trainDF,labelCol,startOfFeats,kCVParam,nSamp,numCores)
	print("   COMPLETE!")

	# Tuning algorithm coefficients and returning the super learner
	print("")
	print("   TUNING SUPER LEARNER & TRAINING COEFFICIENTS")
	print("   ---------------------------------------------------------")
	superLearner = optimizeSuperLearner(choice,paramSet,kCVParam,nSamp,numCores)
	print("   COMPLETE!")

	return superLearner

# --------------------

def optimizeSubmodels(choice,submodels,trainDF,labelCol,startOfFeats,kCVParam,nSamp,numCores):

	# Vectorize the dataframes for training and testing
	train = vectorizeDataFrame(trainDF,labelCol,startOfFeats)

	# Containers for output
	optModels={}
	holdOuts={'label':train['y']}

    # Print progress bar for each model
	printProgressBar(0,len(submodels),prefix='   Progress:',suffix='',length=50)
    		
	# Iterate through submodels of the super learner
	for i in range(0,len(submodels)):

		# Get current algo and probability distrubitions for parameter search
		# Last input (training set size) only used by the multi-layer perceptron
		modelParams = getModelParams(choice,submodels[i],trainDF.shape[1]-startOfFeats,trainDF.shape[0])
		parameters = modelParams['parameters']
		model = modelParams['model']
		outModel = copy.deepcopy(model)

		# Use randomized search for cross-validated hyperparameter optimization
		# Parameter options are sampled with replacement
		# Process using multi-core parallel processing if available
		optimize_pool = Parallel(n_jobs=numCores)(delayed(randSearch)\
			(samp=currIter,choice=choice,searchType="models",train=train,\
			model=model,parameters=parameters,kCVParam=kCVParam) for currIter in range(0,nSamp))
		optimizeResults = mergePoolOutput(choice,"modelBuilding",optimize_pool)

		# Clear the parallel processes after best params are saved
		get_reusable_executor().shutdown(wait=False)

		# Set parameters for the best model, make a copy for generating hold-outs, and train
		# For SVM classification, need a class assignment probability calibration step
		outModel.set_params(**optimizeResults['bestParams'])
		holdOutModel = copy.deepcopy(outModel)
		optModels[submodels[i]] = outModel.fit(train['x'],train['y'])
		if choice == 0 and submodels[i] == "svm":
			holdOutModel = CalibratedClassifierCV(base_estimator=holdOutModel,method="sigmoid")
			optModels[submodels[i]] = CalibratedClassifierCV(base_estimator=optModels[submodels[i]],\
				method="sigmoid",cv="prefit").fit(train['x'],train['y'])
			
		# Choose hold-out prediction method:
		# For classification, class assignment probabilities are used
		if choice == 0:
			predMethod = "predict_proba"
		else:
			predMethod = "predict"

		# Generate hold-outs for the super learner
		preds = cross_val_predict(holdOutModel,X=train['x'],y=train['y'],cv=kCVParam,\
			n_jobs=numCores,method=predMethod)
		if preds.ndim == 1 :
			holdOuts[submodels[i]] = preds
		else:
			for j in range(1,preds.ndim):
				holdOuts[str(submodels[i]+"_"+str(j))] = preds[:,j]

		# Update Progress Bar
		sleep(0.1)
		printProgressBar(i+1,len(submodels),prefix='   Progress:',suffix='',length=50)

	# Compile new dataset with out-of-fold predictions
	newDF = pd.DataFrame.from_dict(holdOuts)

	# Return models with tuned hyperparameters and hold-outs for super learner
	return {'optModels':optModels,'holdOutDF':newDF}

# --------------------

def getModelParams(choice,modelName,numFeats,trainingSize):
    	
	# Definte a unique random number generator
	rng = np.random.RandomState()	
    	
	if modelName=="glm":

		# alpha - constant multipled by the regularization term
		alphas = loguniform(a=0.00001,b=1,scale=1000)

		# tol - tolerance for stopping criterion
		tols = loguniform(a=0.0001,b=1,scale=100)

		# eta0 - initial learning rate (depending on the learning rate type)
		eta0s = loguniform(a=0.001,b=1,scale=1)
		
		# learning_rate - defines the update rule for stochastic gradient descent 
		learning_rates = ["optimal","invscaling","adaptive"]

		# l1_ratio - the mixing parameter for elastic net (l1_ratio = 0: ridge, l1_ratio = 1: lasso)
		l1_ratios = loguniform(a=0.01,b=1,scale=1)

		# n_estimators - number of models to train (bootstrap aggregating)
		n_estimators =  [int(x) for x in np.linspace(start=10,stop=500,num=50)]

		# max_feature - proportion of total features to sample for each estimator (bootstrap aggregating)
		max_features = uniform(loc=0.1,scale=0.9)

		# max_samples - proporation of training set to sample for each estimator (bootstrap aggregating)
		max_samples = uniform(loc=0.1,scale=0.9)

		# Combine parameters
		parameters = {'base_estimator__alpha':alphas,'base_estimator__tol':tols,\
			'base_estimator__learning_rate':learning_rates,'base_estimator__eta0':eta0s,\
			'base_estimator__l1_ratio':l1_ratios,'n_estimators':n_estimators,'max_features':max_features,\
			'max_samples':max_samples}

		# Define model based on classification or regression
		if(choice==0):
			model = BaggingClassifier(base_estimator=SGDClassifier(loss="log",penalty="elasticnet",\
				max_iter=10000,n_jobs=1,random_state=rng,class_weight="balanced"),\
				bootstrap_features=True,n_jobs=1,random_state=np.random.RandomState())
		else:
			model = BaggingRegressor(base_estimator=SGDRegressor(loss="epsilon_insensitive",epsilon=0.0,penalty="elasticnet",\
				max_iter=10000,random_state=rng),\
				bootstrap_features=True,n_jobs=1,random_state=np.random.RandomState())	

	elif modelName=="mlp":
    		
		# alpha - constant multipled by the regularization term
		alphas = loguniform(a=0.00001,b=1,scale=1000)

		# learning_rate_init - initial learning rate (depending on the learning rate type)
		learning_rate_inits = loguniform(a=0.001,b=1,scale=1)

		# tol - tolerance for stopping criterion
		tols = loguniform(a=0.00001,b=1,scale=100)
		
		# solver - learning algorithm for feed-forward and back propogation
		solvers = ["adam","lbfgs"] 

		# activation - activation function used for hidden layer neurons
		activations = ["logistic", "tanh", "relu"]

		# hidden_layer_sizes - number of neurons in the hidden layer (for this iteration, assumes single hidden layer)
		hidden_layer_sizes = [(int(x),) for x in np.linspace(start=10,stop=100,num=10)]

		# n_estimators - number of models to train (bootstrap aggregating)
		# n_estimators has fewer options for the mlp because of computation time to build/fit estimators
		# Any more than ~100 estimators will cause a significiant increase in time complexity
		# that is untenable given our hardware and time constraints
		n_estimators = [int(x) for x in np.linspace(start=10,stop=500,num=50)]

		# max_feature - proportion of total features to sample for each estimator (bootstrap aggregating)
		max_features = uniform(loc=0.05,scale=0.95)

		# max_samples - proporation of training set to sample for each estimator (bootstrap aggregating)
		max_samples = uniform(loc=0.1,scale=0.9)

		# Combine parameters
		parameters = {'base_estimator__hidden_layer_sizes':hidden_layer_sizes, 'base_estimator__alpha':alphas,\
			'base_estimator__tol':tols,'base_estimator__learning_rate_init':learning_rate_inits,\
			'base_estimator__solver':solvers,'base_estimator__activation':activations,\
			'n_estimators':n_estimators,'max_features':max_features,'max_samples':max_samples,}

		# Define model based on classification or regression
		if(choice==0):
			model = BaggingClassifier(base_estimator=MLPClassifier(shuffle=True,max_iter=10000,random_state=rng),\
				bootstrap_features=True,n_jobs=1,random_state=np.random.RandomState())
		else:
			model = BaggingRegressor(base_estimator=MLPRegressor(shuffle=True,max_iter=10000,random_state=rng),\
				bootstrap_features=True,n_jobs=1,random_state=np.random.RandomState())

	elif modelName=="kNN":

		########################################
		# TODO: UPDATE THIS WHEN THERE'S TIME
		########################################

		# Define hyperparameter sampling distributions
		n_neighbors_raw = stats.uniform.rvs(loc=1,scale=round(trainingSize/2),size=5000)

		# Make sure hyperparameters are in the correct range
		n_neighbors = [int(round(x)) for x in n_neighbors_raw]
		ps = [1,2,3,4,5]
		weights = ["uniform","distance"]

		# Combine parameters
		parameters = {'n_neighbors':n_neighbors,'p':ps,'weights':weights}

		# Define model based on classification or regression
		if(choice==0):
			model = KNeighborsClassifier(algorithm="auto")
		else:
			model = KNeighborsRegressor(algorithm="auto")

	elif modelName=="randForest":
    		
		# ccp_alpha - constant multipled by the regularization parameter (Minimal-Cost Complexity Pruning)
		ccp_alphas = loguniform(a=0.00001,b=1,scale=1000)

		# min_samples_split - the minimum number of remaining samples to split a node
		min_samples_split = [int(x) for x in np.linspace(start=2,stop=50,num=48)]
		
		# min_samples_leaf - the minimum number of remaining samples to remain a leaf/terminal node
		min_samples_leaf = [int(x) for x in np.linspace(start=2,stop=50,num=48)]

		# max_depth - how many nodes deep a decision tree can grow
		max_depths = [int(x) for x in np.linspace(start=5,stop=100,num=20)]

		# criterion - node splitting statistic used for decision tree (classification only)
		criterions = ["gini","entropy"]

		# n_estimators - number of models to train (bootstrap aggregating)
		n_estimators = [int(x) for x in np.linspace(start=10,stop=500,num=50)]

		# max_features - proportion of total features to sample for each estimator (bootstrap aggregating)
		max_features = uniform(loc=0.05,scale=0.95)

		# max_samples - proporation of training set to sample for each estimator (bootstrap aggregating)
		max_samples = uniform(loc=0.1,scale=0.9)

		# Combine parameters
		parameters = {'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf,'ccp_alpha':ccp_alphas,\
			'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depths,'max_samples':max_samples}

		# Define model based on classification or regression
		if(choice==0):
			model = RandomForestClassifier(n_jobs=1,random_state=rng,bootstrap=True,class_weight="balanced_subsample")
			parameters['criterion'] = criterions
		else:
			model = RandomForestRegressor(criterion="mae",n_jobs=1,bootstrap=True,random_state=rng)

	elif modelName=="svm":

		# C - 1/lambda (regularization parameter)
		Cs = loguniform(a=0.000001,b=1,scale=100)

		# tol - tolerance for stopping criterion
		tols = loguniform(a=0.0001,b=1,scale=100)

		# epsilon - defines the "epsilon tube", provides some slack (wiggle room) to the decision boundary
		epsilons = loguniform(a=0.000001,b=1,scale=1000)

		# gamma - kernel coefficient when kernel = poly, rbf, or sigmoid
		gammas = loguniform(a=0.0000001,b=1,scale=100)

		# kernel - function for creating a non-linear decision hyperplane		
		kernels = ["poly","rbf","sigmoid"]

		# degree - degree of polynomial for the "poly" kernel
		degrees = [2,3,4,5]

		# n_estimators - number of models to train (bootstrap aggregating)
		n_estimators = [int(x) for x in np.linspace(start=10,stop=500,num=50)]

		# max_feature - proportion of total features to sample for each estimator (bootstrap aggregating)
		max_features = uniform(loc=0.05,scale=0.95)

		# max_samples - proporation of training set to sample for each estimator (bootstrap aggregating)
		max_samples = uniform(loc=0.2,scale=0.8)

		parameters = {'base_estimator__C':Cs,'base_estimator__tol':tols,'base_estimator__degree':degrees,\
			'base_estimator__kernel':kernels,'base_estimator__gamma':gammas,\
			'n_estimators':n_estimators,'max_features':max_features,'max_samples':max_samples}

		# Define model based on classification or regression
		if(choice==0):
			model = BaggingClassifier(base_estimator=SVC(random_state=rng,class_weight="balanced"),\
				bootstrap_features=True,n_jobs=1,random_state=rng)
		else:
			model = BaggingRegressor(base_estimator=SVR(),bootstrap_features=True,n_jobs=1,random_state=rng)
			parameters['base_estimator__epsilon'] = epsilons

	elif modelName=="gradBoost":

		# learning_rate - the amount to shrink the size of each successive tree
		learning_rates = loguniform(a=0.001,b=1,scale=1)

		# subsample - proporation of training set to sample for each estimator (bootstrap aggregating)
		subsamples = uniform(loc=0.1,scale=0.9)

		# ccp_alpha - constant multipled by the regularization parameter (Minimal-Cost Complexity Pruning)
		ccp_alphas = loguniform(a=0.00001,b=1,scale=1000)

		# min_samples_split - the minimum number of remaining samples to split a node
		min_samples_split = [int(x) for x in np.linspace(start=2,stop=50,num=48)]
		
		# min_samples_leaf - the minimum number of remaining samples to remain a leaf/terminal node
		min_samples_leaf = [int(x) for x in np.linspace(start=2,stop=50,num=48)]

		# max_depth - how many nodes deep a tree can grow
		max_depths = [int(x) for x in np.linspace(start=2,stop=10,num=8)]
		
		# loss function to be optimized (different for classification vs. regression)
		losses_class = ["deviance","exponential"]
		losses_regress = ["ls", "lad", "huber", "quantile"]

		# n_estimators - number of models to train (bootstrap aggregating)
		n_estimators = [int(x) for x in np.linspace(start=10,stop=500,num=50)]

		# max_features - proportion of total features to sample for each estimator (bootstrap aggregating)
		max_features = uniform(loc=0.05,scale=0.95)

		# Combine parameters
		parameters = {'learning_rate':learning_rates,'n_estimators':n_estimators,'max_features':max_features, \
			'subsample':subsamples,'max_depth':max_depths,'min_samples_split':min_samples_split,\
			'min_samples_leaf':min_samples_leaf,'ccp_alpha':ccp_alphas}

		# Define model based on classification or regression
		if(choice==0):
			model = GradientBoostingClassifier(criterion="mae",random_state=rng)
			parameters['loss'] = losses_class
		else:
			model = GradientBoostingRegressor(criterion="mae",random_state=rng)
			parameters['loss'] = losses_regress

	elif modelName=="gaussProc":
    		
		########################################
		# TODO: UPDATE THIS WHEN THERE'S TIME
		########################################
    		
		# Define hyperparameter sampling distributions
		n_restarts_optimizer_raw = stats.norm.rvs(loc=50,scale=20,size=5000)
		max_iter_predicts_raw = stats.norm.rvs(loc=50,scale=20,size=5000)
	
		# Make sure hyperparameters are in the correct range
		n_restarts_optimizers = [int(round(i)) for i in n_restarts_optimizer_raw if (int(round(i)) > 0)]
		max_iter_predicts = [int(round(i)) for i in max_iter_predicts_raw if (int(round(i)) > 0)]

		# Kernels obtained from scikit-learn documentation and examples
		# kernel1 = constant + noise
		# kernel2 = constant + RBF + noise
		# kernel3 = constant + Matern + noise
		kernel1 = ConstantKernel() + WhiteKernel()
		kernel2 = ConstantKernel() * RBF() + WhiteKernel()
		kernel3 = ConstantKernel() * Matern() + WhiteKernel()
		kernels = [kernel1,kernel2,kernel3]

		# Combine parameters
		parameters = {'kernel':kernels,'n_restarts_optimizer':n_restarts_optimizers}

		# Define model based on classification or regression
		if(choice==0):
			model = GaussianProcessClassifier()
			parameters['max_iter_predict'] = max_iter_predicts
		else:
			model = GaussianProcessRegressor(normalize_y=True)

	elif modelName=="SL":
    		
		# alpha - constant multipled by the regularization term
		alphas = loguniform(a=0.00001,b=1,scale=1000)

		# tol - tolerance for stopping criterion
		tols = loguniform(a=0.0001,b=1,scale=100)

		# eta0 - initial learning rate (depending on the learning rate type)
		eta0s = loguniform(a=0.001,b=1,scale=1)
		
		# learning_rate - defines the update rule for stochastic gradient descent 
		learning_rates = ["optimal", "invscaling", "adaptive"]

		# l1_ratio - the mixing parameter for elastic net (l1_ratio = 0: ridge, l1_ratio = 1: lasso)
		l1_ratios = loguniform(a=0.001,b=1,scale=1)

		# Combine parameters
		parameters = {'alpha':alphas,'tol':tols,'learning_rate':learning_rates, \
			'eta0':eta0s,'l1_ratio':l1_ratios}

		# Define model based on classification or regression
		if(choice==0):
			model = SGDClassifier(loss="log",penalty="elasticnet",max_iter=100000,n_jobs=1,
				random_state=rng,fit_intercept=False,class_weight="balanced")
		else:
			model = SGDRegressor(loss="epsilon_insensitive",epsilon=0.0,penalty="elasticnet",max_iter=100000,
				random_state=rng,fit_intercept=False)
    		
	elif modelName=="optics":
    	
		# eps - maximum distance between samples to be considered in the neighborhood of eachother (dbscan method only)
		eps = uniform(loc=0.0,scale=1.0)

		# xi - minimum steepness on the reachability plot that constitutes a cluster boundary (xi method only)
		xis = uniform(loc=0.0,scale=1.0)

		# cluster_method - extraction method for cluster labels (based on 'reachability' statistic and feature order)
		cluster_methods = ["xi","dbscan"]

		# predecessor_correction - correction based on previous clusters identified by OPTICS
		predecessor_correction = [True,False]

		# Combine parameters
		parameters = {'cluster_method':cluster_methods,'eps':eps,'xi':xis,'predecessor_correction':predecessor_correction}

		# Define clustering model
		model = OPTICS(n_jobs=1,min_samples=2,min_cluster_size=2,metric="precomputed")
	
	return {'model':model,'parameters':parameters}

# --------------------

def randSearch(samp,choice,searchType,train,model,parameters,kCVParam):
	
	# Get random sample of parameters to try, set them
	randParams = getRandParams(parameters)
	cvModel = copy.deepcopy(model)
	cvModel.set_params(**randParams)

	# Run search for submodel parameters
	if searchType == "models":
    		
		# Define hold-out container
		holdOuts = {}

		# Designate splitter for cross-validation
		if choice == 0:
			skf = StratifiedKFold(n_splits=kCVParam,shuffle=True,random_state=np.random.RandomState())
		else:
			skf = KFold(n_splits=kCVParam,shuffle=True,random_state=np.random.RandomState())

		# Test parameters using cross-validation
		preds = cross_val_predict(cvModel,X=train['x'],y=train['y'],cv=skf,n_jobs=1,method="predict",pre_dispatch=None)

		# Score the predictions (and get prediction probabilities for classification)
		if choice == 0:
			score = balanced_accuracy_score(train['y'],preds)
		else:
			score = mean_absolute_error(train['y'],preds)
    		
		# Return the mean score and parameters used
		return {'score':score,'params':randParams}
    		
	# Run search for clustering parameters
	elif searchType == "clustering":

		# Run the clustering and get the predicted cluster labels
		clustLabels = cvModel.fit_predict(train)

		# Return the silhouette coefficient for this round of clustering
		return {'silCoef':getClusteringScore(clustLabels,train),'clustLabels':clustLabels}

# --------------------

def getRandParams(params):
    	
	# Choose random parameters from each key in dict
	outDict = {}
	for key in params:
		rng = np.random.RandomState()

		# Get random parameter from a list
		if isinstance(params[key],list):
			randInt = int(rng.randint(0,len(params[key]),1))
			outDict[key] = params[key][randInt]

		# Get random parameter from a distribution
		else:
			outDict[key] = float(params[key].rvs(size=1,random_state=rng))

	return outDict

# --------------------

def optimizeSuperLearner(choice,paramSet,kCVParam,nSamp,numCores):

	# Vectorize evaluation dataset
	holdOutVec = vectorizeDataFrame(paramSet['holdOutDF'],0,1)

	# Uses same linear model framework (trained with stochastic gradient descent) as
	# the GLM model - elastic net
	sdgCoef = getModelParams(choice,"SL",paramSet['holdOutDF'].shape[1]-1,paramSet['holdOutDF'].shape[0])
	model = sdgCoef['model']
	params = sdgCoef['parameters']
	outModel = copy.deepcopy(model)

	# Use randomized search for cross-validated hyperparameter optimization
	# Parameter options are sampled with replacement
	# Process using multi-core parallel processing if available
	optimize_pool = Parallel(n_jobs=numCores)(delayed(randSearch)\
		(samp=currIter,choice=choice,searchType="models",train=holdOutVec,\
		model=model,parameters=params,kCVParam=kCVParam) for currIter in range(0,nSamp))
	optimizeResults = mergePoolOutput(choice,"modelBuilding",optimize_pool)

	# Clear the parallel processes after best params are saved
	get_reusable_executor().shutdown(wait=False)

	# Train and save the optimized model
	outModel.set_params(**optimizeResults['bestParams'])
	outModel.fit(holdOutVec['x'],holdOutVec['y'])

	# Create the new Super Learner and return
	return {'models':paramSet['optModels'],'SL':outModel}

# ================================================== MODEL EVALUATION ===================================================

def evaluateSuperLearner(SL,choice,DFs,batchNum,numSimsPI,labelCol,startOfFeats,numCores):
    	
	print("")
	print("   EVALUATING SUPER LEARNER & SUBMODEL PERFORMANCES")
	print("   ---------------------------------------------------------")

	# Standardize current batch (to mean/SD of current training set)
	normBatch = standardizeBatch(DFs,labelCol,startOfFeats)

	# Get dataset to evaluate the model and vectorize
	testDF = normBatch['testDF']
	trainDF = normBatch['trainDF']	
	testVec = vectorizeDataFrame(testDF,labelCol,startOfFeats)
	trainVec = vectorizeDataFrame(trainDF,labelCol,startOfFeats)

	# Get super learner preds
	testPreds = predictWithSuperLearner(choice,SL,testVec)
	trainPreds = predictWithSuperLearner(choice,SL,trainVec)

	# Get scores for the super learner and submodels on the test and training data
	sub_scores_test = {'SL':getScores(choice,testVec['y'],testPreds['preds'],testPreds['probs'])}
	sub_scores_train = {'SL':getScores(choice,trainVec['y'],trainPreds['preds'],trainPreds['probs'])}
	for key in SL['models']:
		if choice == 0:
			testProbs = SL['models'][key].predict_proba(testVec['x'])
			trainProbs = SL['models'][key].predict_proba(trainVec['x'])
			# Output for binary vs. multiclass classification
			if len(set(trainVec['y'])) == 2:
				testProbs = testProbs[:,1]
				trainProbs = trainProbs[:,1]
		else:
			testProbs = 0.0
			trainProbs = 0.0
		sub_scores_test[key] = getScores(choice,testVec['y'],SL['models'][key].predict(testVec['x']),testProbs)
		sub_scores_train[key] = getScores(choice,trainVec['y'],SL['models'][key].predict(trainVec['x']),trainProbs)

	# Combine scores
	all_scores = {'test':sub_scores_test,'train':sub_scores_train}

	# Calculate probability values for scores based on a null distribution
	all_ps = calcAboveChancePs(choice,numSimsPI,SL,normBatch,all_scores,labelCol,startOfFeats)
	print("   COMPLETE!")

	# Print/Return super learner coeficients and submodel scores and probabilities
	return {'coefsOut':printSLCoefReport(choice,SL,trainVec['y'],batchNum),\
		'test':printScoreReport(sub_scores_test,all_ps['test'],"test",batchNum),\
		'train':printScoreReport(sub_scores_train,all_ps['train'],"train",batchNum)}

# --------------------

def predictWithSuperLearner(choice,superLearner,evalVec):
    
	# Define container for predictions
	preds = {'label':evalVec['y']}
	predsDim = 1

	# Get hold-out predictions for the super learner
	for key in superLearner['models']:
		if choice == 0:
			currPreds = superLearner['models'][key].predict_proba(evalVec['x'])
			if currPreds.ndim > 1:
				predsDim = currPreds.ndim
				for j in range(1,predsDim):
					preds[str(key)+"_"+str(j)] = currPreds[:, j]
		else:
			currPreds = superLearner['models'][key].predict(evalVec['x'])
			preds[key] = currPreds

	# Create new dataframe of predictions, vectorize, and scale
	predVec = vectorizeDataFrame(pd.DataFrame.from_dict(preds),0,1)
	
	# Feed predictions into super learner and make final prediction
	if choice == 0:
		probs = superLearner['SL'].predict_proba(predVec['x'])
		# Output for binary vs. multiclass classification
		if len(set(evalVec['y']))==2:
			probs = probs[:,1]
	else:
		probs = 0.0
	return {'preds':superLearner['SL'].predict(predVec['x']),'probs':probs}

# --------------------

def calcAboveChancePs(choice,numSimsPI,SL,normBatch,allScores,labelCol,startOfFeats):

	# Define containers for probability statistics
	pVals_out_test = {}
	pVals_out_train = {}

	# Evaluate the current model's scores versus chance performance
	# Simulations: sample labels with replacement
	eval_pool = Parallel(n_jobs=numCores)(delayed(getNullPerfDists)(currIter=currIter,\
		modelNames=list(allScores['test'].keys()),choice=choice,SL=SL,normBatch=normBatch,labelCol=labelCol,\
		startOfFeats=startOfFeats) for currIter in range(0,numSimsPI))
	eval_merge = mergePoolOutput(choice,"nullScores",eval_pool)

	# Clear the parallel processes best params are saved
	get_reusable_executor().shutdown(wait=False)

	# Generate probability values for the super learner and submodel scores
	for key in allScores['test']:
    		
		# Define more stats ontainers
		pVals_out_test[key] = {}
		pVals_out_train[key] = {}

		for key1 in allScores['test'][key]:
			if choice == 0 or str(key1) == "r" or str(key1) == "r2":
				pVals_out_test[key][key1] = calcPFromDist(allScores['test'][key][key1],\
					"greater than",eval_merge['test'][key][key1])
				pVals_out_train[key][key1] = calcPFromDist(allScores['train'][key][key1],\
					"greater than",eval_merge['train'][key][key1])
			else:	
				pVals_out_test[key][key1] = calcPFromDist(allScores['test'][key][key1],\
					"less than",eval_merge['test'][key][key1])
				pVals_out_train[key][key1] = calcPFromDist(allScores['train'][key][key1],\
					"less than",eval_merge['train'][key][key1])

	return {'test':pVals_out_test,'train':pVals_out_train}

# --------------------

def getNullPerfDists(currIter,modelNames,choice,SL,normBatch,labelCol,startOfFeats):
    		
	# Container to return
	toReturn = {'test':{},'train':{}}

	# Get scores for this simulation for the SL and each submodel
	for modelName in modelNames:

		# Make copy of the data
		permute_testDF = copy.deepcopy(normBatch['testDF'])
		permute_trainDF = copy.deepcopy(normBatch['trainDF'])

		# Define a new random number generator and permute label column for test and train DFs
		rng1 = np.random.RandomState()
		rng2 = np.random.RandomState()

		permute_testDF[permute_testDF.columns[labelCol]] = \
			rng1.choice(permute_testDF[permute_testDF.columns[labelCol]],\
			replace=False,size=permute_testDF.shape[0])
		permute_trainDF[permute_trainDF.columns[labelCol]] = \
			rng2.choice(permute_trainDF[permute_trainDF.columns[labelCol]],\
			replace=False,size=permute_trainDF.shape[0])

		# Get vectorized data and evaluate null performance
		perSets_test = vectorizeDataFrame(permute_testDF,labelCol,startOfFeats)
		perSets_train = vectorizeDataFrame(permute_trainDF,labelCol,startOfFeats)

		# Get differences difference in model performance
		if modelName == "SL":
			testPreds = predictWithSuperLearner(choice,SL,perSets_test)
			trainPreds = predictWithSuperLearner(choice,SL,perSets_train)
			toReturn['test'][modelName] = getScores(choice,perSets_test['y'],testPreds['preds'],testPreds['probs'])
			toReturn['train'][modelName] = getScores(choice,perSets_train['y'],trainPreds['preds'],trainPreds['probs'])
		else:
			if choice == 0:
				testProb = SL['models'][modelName].predict_proba(perSets_test['x'])
				trainProb = SL['models'][modelName].predict_proba(perSets_train['x'])
				# Output for binary vs. multiclass classification
				if len(set(perSets_train['y'])) == 2:
					testProb = testProb[:,1]
					trainProb = trainProb[:,1]
			else:
				testProb = 0.0
				trainProb = 0.0
			toReturn['test'][modelName] = getScores(choice,perSets_test['y'],\
				SL['models'][modelName].predict(perSets_test['x']),testProb)
			toReturn['train'][modelName] = getScores(choice,perSets_train['y'],\
				SL['models'][modelName].predict(perSets_train['x']),trainProb)

	return toReturn

# ================================================== FEATURE LEARNING ===================================================

def runFeatureLearning(SL,choice,DFs,batchNum,nSamp,numSimsPI,labelCol,startOfFeats,numCores):
    
	print("")
	print("   CLUSTERING FEATURES TO AVOID MULTICOLLINEARITY")
	print("   ---------------------------------------------------------")

	# Standardize current batch (to mean/SD of current training set)
	normBatch = standardizeBatch(DFs,labelCol,startOfFeats)

	# Use training set to determine feature importance scores
	trainDF = normBatch['trainDF']

	# Vectorize the training data
	trainVec = vectorizeDataFrame(trainDF,labelCol,startOfFeats)

	# Cluster features so that related features are altered together
	clusterLabels = getFeatureClusters(trainVec,labelCol,startOfFeats,nSamp,numCores)
	print("   "+ str(len(set([i for i in clusterLabels if i != -1])))+" unique clusters found.")
	print("   "+str(round((list(clusterLabels).count(-1)/len(clusterLabels))*100,2))+\
		"% of features do not belong to a cluster.")

	# Get batches of features (circumventing multicollinearity) to test
	featBatches = getFeatureBatches(clusterLabels)

	print("   COMPLETE!")
	print("")
	print("   USING PERMUTATION TO CALCULATE FEATURE IMPORTANCE")
	print("   ---------------------------------------------------------")

	# Get the best set of scores for the full training set
	SLpreds = predictWithSuperLearner(choice,SL,trainVec)
	bestScores = getScores(choice,trainVec['y'],SLpreds['preds'],SLpreds['probs'])
	if choice == 0:
		bestScore = bestScores['balAcc']
	else:
		bestScore = bestScores['MAE']
	
	# Calculate feature importance score for each feature in the testDF
	# Simulations: sample feature values with replacement and re-calculate performance
	# Process using multi-core parallel processing if available
	featImp_pool = Parallel(n_jobs=numCores)(delayed(calcFeatureImportance)(featIndex=currBatch,choice=choice,\
		SL=SL,bestScore=bestScore,batches=featBatches,numSimsPI=numSimsPI,testDF=trainDF,labelCol=labelCol,
		startOfFeats=startOfFeats) for currBatch in list(featBatches.keys()))
	featImp_out = mergePoolOutput(choice,"featImp",featImp_pool)

	# Clear the parallel processes once scores are compiled
	get_reusable_executor().shutdown(wait=False)

	# Format the output from pool
	featImp_dic = {}
	featIndices = sorted([int(key) for key in featImp_out])
	for feat in featIndices:
		featImp_dic[trainDF.columns[feat+startOfFeats]]=featImp_out[str(feat)]
	
	print("   COMPLETE!")

	# Return feature importance scores + probability values relative to true performance
	return saveFLReport(clusterLabels,featImp_dic,batchNum)

# --------------------

def getFeatureClusters(trainVec,labelCol,startOfFeats,samp,numCores):

	# Get spearman rho correlation matrix (absolute value)
	rhoMat, pOut = spearmanr(trainVec['x'])
	distRhoMat = 1 - np.absolute(rhoMat)

	# Get clustering model and hyperparameters
	optics = getModelParams(0,"optics",len(trainVec['x'][0]),len(trainVec['x']))

	# Process using multi-core parallel processing if available
	clust_pool = Parallel(n_jobs=numCores)(delayed(randSearch)\
		(samp=currIter,choice=0,searchType="clustering",train=distRhoMat,\
		model=optics['model'],parameters=optics['parameters'],kCVParam=1) for currIter in range(0,samp))
	clusterResults = mergePoolOutput(0,"clustering",clust_pool)

	# Clear the parallel processes once scores are compiled
	get_reusable_executor().shutdown(wait=False)

	# Return the clustering labels with the maximum silhouette coefficient
	return clusterResults['clustLabels'][np.argmax(clusterResults['silCoef'])]

# --------------------

def getClusteringScore(clustLabels,trainMat):

	# Get clusters to score and define outputs
	clusts = np.unique(clustLabels)
	clustsToUse = [x for x in clusts if x != -1]

	# Skip if no clusts are obtained with given parameteres
	if clustsToUse:

		# Container for silhouette scores
		silScores = [0.0]*len(trainMat)

		# Get features belonging to the cluster
		for feat1 in range(0,len(trainMat)):
				
			# if feature doesn't belong to a cluster, silhouette score = 0
			if clustLabels[feat1] == -1:
				silScores[feat1] = 0.0

			# else, calculate the silhouette score for the point:
			# Peter J. Rousseeuw (1987).
			# "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis".
			# Computational and Applied Mathematics. 20: 53–65. doi:10.1016/0377-0427(87)90125-7
			else:
				withinDists, betweenDists = ([], )*2
				for feat2 in range(0,len(trainMat)):
					if feat1 != feat2:
						if clustLabels[feat1] == clustLabels[feat2]:
							withinDists.append(trainMat[feat1][feat2])
						else:
							betweenDists.append(trainMat[feat1][feat2])
				withinStat = np.mean(withinDists)
				betweenStat = np.min(betweenDists)
				if withinStat > betweenStat:
					silScores[feat1] = (betweenStat/withinStat) - 1
				elif betweenStat > withinStat:
					silScores[feat1] = 1 - (withinStat/betweenStat)
				else:
					silScores[feat1] = 0.0

		# Return the silhouette coefficient for this round of clustering
		return np.mean(silScores)

	else:
			
		# Return the silhouette coefficient = 0 if no clusters identified
		return 0.0

# --------------------

def getFeatureBatches(clustLabels):
    
	# Output Dictionary
	featBatches = {}
	batchKey = 1

	# Keep track of which features already have a cluster
	usedFeats = list([])

	# Iterate features
	for i, feat in enumerate(clustLabels):
    		
		# If feature hasn't been checked
		if not (i in usedFeats):
    			
			# No cluster for this feature, add it to checked
			if feat == -1:
				featBatches[str(batchKey)] = i
				usedFeats.append(i)
				batchKey += 1

			# Cluster exists
			# Find others in the cluster, add it (and others) to checked
			else:
				currFeats = list([])
				currFeats.append(i)
				usedFeats.append(i)
				for j, nextFeat in enumerate(clustLabels):
					if not (j in usedFeats):
						if nextFeat == feat:
							currFeats.append(j)
							usedFeats.append(j)
				featBatches[str(batchKey)] = currFeats
				batchKey += 1

	return featBatches

# --------------------

def calcFeatureImportance(featIndex,choice,SL,bestScore,batches,numSimsPI,testDF,labelCol,startOfFeats):
    	
	# Get the current batch of features to work on (may be only one)
	currFeats = batches[str(featIndex)]
	toReturn ={}

	# Container for permuted scores
	impScoreDist = [0.0]*numSimsPI

	# Iterate through rounds of permutation - get mean importance score
	for sim in range(0,numSimsPI):
    		
		# Make copies of the data
		permuteDF = copy.deepcopy(testDF)

		# Replace the feature/s with the permutation leaving all other features the same
		if isinstance(currFeats,int):
			rng = np.random.RandomState()
			permuteDF[permuteDF.columns[currFeats+startOfFeats]] = \
				rng.choice(permuteDF[permuteDF.columns[currFeats+startOfFeats]],\
				replace=False,size=permuteDF.shape[0])
		else:
			for feat in currFeats:
				rng = np.random.RandomState()
				permuteDF[permuteDF.columns[feat+startOfFeats]] = \
					rng.choice(permuteDF[permuteDF.columns[feat+startOfFeats]],\
					replace=False,size=permuteDF.shape[0])

		# Vectorize the permuted data
		permuteVec = vectorizeDataFrame(permuteDF,labelCol,startOfFeats)

		# Get scores with the feature/feature set permutation
		SLpreds = predictWithSuperLearner(choice,SL,permuteVec)
		impScores = getScores(choice,permuteVec['y'],SLpreds['preds'],SLpreds['probs'])

		# Save scores
		if choice == 0:
			impScoreDist[sim] = impScores['balAcc']
		else:
			impScoreDist[sim] = impScores['MAE']

	# Return mean change in performance and probability value relative to the best score
	if isinstance(currFeats,int):
		if choice == 0:
			toReturn[str(currFeats)] = [bestScore - np.mean(impScoreDist),\
				calcPFromDist(bestScore,"greater than",impScoreDist)]
		else:
			toReturn[str(currFeats)] = [np.mean(impScoreDist) - bestScore,\
				calcPFromDist(bestScore,"less than",impScoreDist)]
	else:
		for feat in currFeats:
			if choice == 0:
				toReturn[str(feat)] = [(bestScore - np.mean(impScoreDist))/len(currFeats),\
					calcPFromDist(bestScore,"greater than",impScoreDist)]
			else:
				toReturn[str(feat)] = [(np.mean(impScoreDist) - bestScore)/len(currFeats),\
					calcPFromDist(bestScore,"less than",impScoreDist)]
    			
	return toReturn

# -----------------------------------------------------------------------------------------------------------------------
# MAIN:
# -----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	print("")
	print("-----------------------------------------------------------------------------")
	print("  FEATURE LEARNING using STACKED GENERALIZATION and PERMUTATION IMPORTANCE   ")
	print("-----------------------------------------------------------------------------")

	# Read in all the args and data
	readIn = check_args_read_data(sys.argv)
	allData = readIn['df'] # Dataset as a Pandas dataframe
	labelCol = 1 # Column index for the variable-to-predict in allData (constant)
	startOfFeats = 2 # Column index for the start of features in allData (constant)
	prefix = readIn['prefix'] # Output file(s) prefix
	numSims = readIn['numSims'] # Number of train/test batches
	testProp = readIn['testProp'] # Proportion of data to hold aside for evaluating model performance
	kCVParam = readIn['kCVParam'] # Number of folds for cross-validation in randomized parameter search
	nSampling = readIn['nSampling'] # Rounds of sampling for randomized parameter search
	numSimsPI = readIn['numSimsPI'] # Number of permutation samples for labels (evaluation) and features (importance)
	numCores = readIn['numCores'] # Number of cores to use for parallel processing
	if numCores == -1:
		numCores = mp.cpu_count()

	# Get choice to perform either regression or classification
	choice = 1
	if(allData.dtypes[labelCol]=="int64"):
		if len(allData[allData.columns[labelCol]].unique().tolist()) < 5:
			choice = 0
			print("   Using CLASSIFICATION to PREDICT:",str(list(allData.columns.values)[labelCol]),"\n")
		else:
			allData[allData.columns[labelCol]] = allData[allData.columns[labelCol]]*1.0
			print("   Using REGRESSION to PREDICT:",str(list(allData.columns.values)[labelCol]),"\n")
	else:
		print("   Using REGRESSION to PREDICT:",str(list(allData.columns.values)[labelCol]),"\n")

	# Get submodel algorithms to use in the super learner
	# "randForest" = Random Forest Classification/Regression
	# "svm" = Support Vector Classification/Regression with Bootstrap Aggregation
	# "glm" = Multi-class Logistic and Linear Regression with Bootstrap Aggregation
	# "mlp" = Multilayer Perceptron (single hidden layer) Classification/Regression with Bootstrap Aggregation
	# "gradBoost" = Gradient Boosting Classification/Regression
	modelsToUse = ["svm","mlp","gradBoost","randForest","glm"]

	# Other available models not suitable for this problem (...with more to come):
	# "gaussProc" = Gaussian Process Regression - too many features for the computational resources available 
	# "kNN" = k-Nearest Neighbors Classification/Regression - too much collinearity of features

	# Get training/testing batches
	batches_raw = getKFoldBatches(choice,labelCol,startOfFeats,allData,numSims,testProp)

	# Define containers for output model and feature importance statistics
	SLCoefsOut, testOut, trainOut, FLOut = ([], )*4

	# Iterate through batches
	for count, key in enumerate(batches_raw):

		# Get the current time
		batchTime = datetime.now()

		# Optimize submodel and super learner hyperparameters and train
		SL_out = buildSuperLearner(choice,batches_raw[key],modelsToUse,labelCol,startOfFeats,kCVParam, \
			nSampling,numCores)

		# Evaluate the super learner
		eval_out = evaluateSuperLearner(SL_out,choice,batches_raw[key],count+1,numSimsPI,labelCol,startOfFeats,numCores)

		# Save coefficients and stats for this batch
		SLCoefsOut.append(eval_out['coefsOut'])
		
		# Output evaluation stats for the test and training sets
		testOut.append(eval_out['test'])
		trainOut.append(eval_out['train'])

		print("")
		print("   Model Building + Evaluation Time: "+str(datetime.now()-batchTime))
		print("")

		# Use the super learner to conduct feature learning using permutation importance
		# Output feature importance statistics
		FL_out = runFeatureLearning(SL_out,choice,batches_raw[key],count+1,nSampling,numSimsPI,labelCol,\
			startOfFeats,numCores)

		# Save feature learning stats for this batch
		FLOut.append(FL_out)

		print("")
		print("   Total Processing Time for Batch "+str(count+1)+": "+str(datetime.now()-batchTime))
		print("")

	# Create output dataframe for evaluation and feature importance distributions
	saveDFtoFile(pd.concat(SLCoefsOut,ignore_index=True,sort=False),\
		str(prefix+"_SLCoefs_"+allData.columns[labelCol]+"_cv"+str(kCVParam)+"_simsPI"+str(numSimsPI)+".csv"))
	saveDFtoFile(pd.concat(testOut,ignore_index=True,sort=False),\
		str(prefix+"_modelEvalTest_"+allData.columns[labelCol]+"_cv"+str(kCVParam)+"_simsPI"+str(numSimsPI)+".csv"))
	saveDFtoFile(pd.concat(trainOut,ignore_index=True,sort=False),\
		str(prefix+"_modelEvalTrain_"+allData.columns[labelCol]+"_cv"+str(kCVParam)+"_simsPI"+str(numSimsPI)+".csv"))
	saveDFtoFile(pd.concat(FLOut,ignore_index=True,sort=False),\
		str(prefix+"_featImp_"+allData.columns[labelCol]+"_cv"+str(kCVParam)+"_simsPI"+str(numSimsPI)+".csv"))

	print("-----------------------------------------------------------------------------")
	print("                            ALL ANALYSES COMPLETE!                           ")
	print("               Total Execution Time: "+str(datetime.now() - startTime)+"          ")
	print("-----------------------------------------------------------------------------")
	print("")

# -----------------------------------------------------------------------------------------------------------------------
# END MAIN
# -----------------------------------------------------------------------------------------------------------------------
