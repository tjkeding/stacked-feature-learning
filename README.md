# stacked-feature-learning
A tool to build a stacked generalizer, evaluate its predictions, and interrogate informative features using permutation importance.

## Implementation:

**Dependencies**: Python3, Joblib, SciKit-Learn, Numpy, Pandas, SciPy

**Input**:  
```
python3 stacked-feature-learning.py <dataFileName> <outputPrefix> <numSims> <testProp> <kCVParam> <nSampling> <numSimsPI> <numCores>
```  

- *\<dataFileName\>*: file path to the data file in CSV format
- *\<outputPrefix\>*: string for prefix to append to all output files 
- *\<numSims\>*: number of different train/test batches to evaluate predictions and features over
- *\<testProp\>*: proportion of instances in data file to use for the held-out test/evaluation set
- *\<kCVParam\>*: number of folds to use when optimizing models using randomized parameter search cross validation
- *\<nSampling\>*: number of sampling iterations to using during randomized parameter search
- *\<numSimsPI\>*: number of random permutations to use when calculating permutation importance (also used to calculate p-values during model evaluation)  
- *\<numCores\>*: number of cores to use for parallel computing  
   
Example:  
```
python3 stacked-feature-learning.py data.csv featureLearning/FL_batches50_sims1000 50 0.15 5 1000 1000 50
```

Notes on CSV file format:  
- Rows are instances
- Column 1 contains the instance ID
- Column 2 contains the label to be predicted (regression: real, continuous; classification: positive integers with 0 = base class)
- Columns 3...N contain the feature values, with the column header containing the feature label (assumed already preprocessed)
  
**Output Files**:  
{outputPrefix}\_SLCoefs\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Stacked generalizer (aka "Super Learner") coefficients learned for each train/test batch and submodel
- Rows: batch/submodel combinations
- Columns: batch number (BATCH); submodel (MODEL); coefficient (COEF)

{outputPrefix}\_modelEvalTest\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Model evaluation metrics for each batch test set
- Rows: batch/model/statistic combinations
- Columns: batch number (BATCH); model names: including "SL" (super learner) and each submodel (MODEL); statistic: For classification tasks, this includes balanced accuracy (optimized), raw accuracy, area under the ROC curve, f1 score, precision, recall, and average precision. For regression tasks, this includes mean absolute error (optimized), mean squared error, root mean squared error, correlation coefficient (r), and coefficient of determination (r2) (STAT); score for the statistic (SCORE); probability of the statistic: score >= chance expectations for mean absolute error, mean squared error, root mean squared error, score <= chance expectations for the rest (P)

{outputPrefix}\_modelEvalTrain\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Model evaluation metrics for each batch training set
- Rows: batch/model/statistic combinations
- Columns: batch number (BATCH); model names: including "SL" (super learner) and submodels (MODEL); statistic: For classification tasks, this includes balanced accuracy (optimized), raw accuracy, area under the ROC curve, f1 score, precision, recall, and average precision. For regression tasks, this includes mean absolute error (optimized), mean squared error, root mean squared error, correlation coefficient (r), and coefficient of determination (r2) (STAT); score for the statistic (SCORE); probability of the statistic: score >= chance expectations for mean absolute error, mean squared error, root mean squared error, score <= chance expectations for the rest (P)

{outputPrefix}\_featImp\_{PRED_VAR_NAME}\_cv{kCVParam}
- Feature importance statistics based on permutation importance
- Rows: batch/feature combinations
- Columns: batch number (BATCH); feature names (FEAT); feature cluster: to avoid issues with multicollinear features during permutation importance, feature clustering is performed. Integer indicating which cluster the feature belonged to for that batch (-1 = no cluster) (CLUST); permutation importance score: mean change in super learner perfomance across rounds of permutation (features belonging to a cluster are always permuted simultaneously and the assigned score for each is scaled to the number of features in that cluster) (SCORE); probability of the score: probability of performance change relative to the best (optimized, non-permuted) score (P) 

## Analysis Summary:
The analysis pipeline consists of:

### Building the Stacked Generalizer (Super Learner)
------
1) Multiple machine learning algorithms\* (submodels) are tuned using randomized parameter search with cross-validation and trained on the current batches' full training set. Submodels currently available include elastic net trained with stochastic gradient decsent (glm), non-linear support vector machine (svm), multilayer perceptron (mlp), random forest (randForest), and gradient boosting machine (gradBoost). All submodels aside from the random forest and gradient boosting machines are built in a bagging (bootstrap aggregation) ensemble to ensure comparable performance to random forest and gradient boosting. All submodels are optimized using mean absolute error (regression) or balanced accuracy (classification).
2) Prediction submodels are aggregated into a super learner<sup>1,2</sup>, a linear combination of submodel predictions optimizing the same loss function. This is implemented as an elastic net model with no intercept and trained with stochastic gradient descent. The super learner is tuned using randomized paramter search and trained on the cross validation hold-outs from the full training sample. 

\*All algorithms are implemented using *SciPy*, *Statsmodels*, and *SciKit-Learn*, with many more to come.

### Evaluating Super Learner and Submodel Performances
------
1) The optimized and trained super learner and submodels are scored on the full training set and test set. The super learner coefficients and all scores are printed to the command line and saved.
2) To determine above-chance super learner and submodel performances, permutation testing is used. Here, {nSimsPI} number of permutations of the label column are created and scored with the super learner and each submodel. The best (non-permuted) score is compared to this distribution of permuted performances and probability values are printed to the command line and saved.

### Feature Importance Analysis using Permutation Importance
------
1) Permutation importance is a univariate feature importance metric that represents the decrease in model performance when the relationship between that feature's values and other features/the label are disturbed. This can be interpretted as the "influence" of that feature alone on making accurate predictions through the model.
2) The permutation importance approach is limited by multicollinearity between features (i.e. importance is underestimated for features where the model continues to get the useful information from other related features). To circumvent this issue, an absolute Spearman rho correlation matrix is generated for the features. These pairwise relationships are input as the distance matrix to an OPTICS clustering algorithm (implemented in SciKit-Learn). OPTICS parameters are optimized through randomized parameter search and scored using the Dunn Index<sup>3</sup>. Features are either assigned to a cluster or left on their own (signified by cluster = -1), depending on the results of the clustering.
3) Similar to permutation testing, {nSimsPI} rounds of permuting each feature column are used to disturb feature/label relationships. If a feature belongs to a cluster, all features in that cluster are permuted simultaneously. The mean change in performance of the super learner is assigned to that feature as its importance (for clustered features, this score is scaled to the number of features in that cluster). The best (non-permuted feature) score is then compared to these permuted feature scores to generate probability values for the importance score. All scores/statistics are saved to the output file.

More detailed information to come! For specific questions, contact tjkeding@gmail.com.


## License
© Taylor J. Keding, 2020. Licensed under the General Public License v3.0 (GPLv3).
This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute (but please cite this repository when appropriate; contact tjkeding@gmail.com for more details).


## Acknowledgements & Contributions
Special thanks to Justin Russell Ph.D., Josh Cisler Ph.D., and Jerry Zhu Ph.D. (University of Wisconsin-Madison) for their input on implementation and best-practices. stacked-feature-learning is open for improvements and maintenance. Your help is valued to make the package better for everyone!


# References
**(1)** van der Laan, M.J., Polley, E.C., & Hubbard, A.E. Super learner. *Statistical Applications in Genetics and Molecular Biology*. 6,25 (2007). DOI: 10.2202/1544-6115.1309

**(2)** Naimi, A.I. & Balzer, L.B. Stacked Generalization: An Introduction to Super Learning. *European Journal of Epidemiology*. 33, 459–464 (2018). DOI: 10.1007/s10654-018-0390-z

**(3)** Dunn, J.C. Well-Separated Clusters and Optimal Fuzzy Partitions. *Journal of Cybernetics*. 4(1): 95–104 (1974). DOI: 10.1080/01969727408546059
