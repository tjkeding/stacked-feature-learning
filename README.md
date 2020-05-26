# stacked-feature-learning
A tool to build a stacked generalizer, evaluate it's predictions, and interogate informative features using permutation importance.

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
- Rows should be one per instance/subject
- Column 1 (index 0) contains the instance/subject ID
- Column 2 contains the label to be predicted (regression: real, continuous; classification: positive integers with 0 = base class)
- Columns 3...N contain the feature values, with the column header containing the feature label (assumed already preprocessed)
  
**Output**:  
{prefix}\_SLCoefs\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Stacked generalizer (aka "Super Learner") coefficients learned for each train/test batch and submodel
- Rows: batch/submodel combinations
- Columns: batch number (BATCH); submodel (MODEL); coefficient (COEF)

{outputPrefix}\_modelEvalTest\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Model evaluation metrics for each batch test set
- Rows: batch/model/statistic combinations
- Columns: batch number (BATCH); model names, including "SL" (super learner) and submodels (MODEL); statistic: for classification tasks, this includes balanced accuracy (optimized), raw accuracy, area under the ROC curve, f1 score, precision, recall, and average precision. For regression tasks, this includes mean absolute error (optimized), mean squared error, root mean squared error, correlation coefficient (r), and coefficient of determination (r2) (STAT); score for the statistic (SCORE); probability of the statistic: score <= chance expectations for mean absolute error, mean squared error, root mean squared error, score >= chance expectations for the rest (P)

{outputPrefix}\_modelEvalTrain\_{PRED_VAR_NAME}\_cv{kCVParam}\_simsPI{numSimsPI}.csv
- Model evaluation metrics for each batch training set
- Rows: batch/model/statistic combinations
- Columns: batch number (BATCH); model names, including "SL" (super learner) and submodels (MODEL); statistic: for classification tasks, this includes balanced accuracy (optimized), raw accuracy, area under the ROC curve, f1 score, precision, recall, and average precision. For regression tasks, this includes mean absolute error (optimized), mean squared error, root mean squared error, correlation coefficient (r), and coefficient of determination (r2) (STAT); score for the statistic (SCORE); probability of the statistic: score <= chance expectations for mean absolute error, mean squared error, root mean squared error, score >= chance expectations for the rest (P)

{outputPrefix}\_featImp\_{PRED_VAR_NAME}\_cv{kCVParam}
- Feature importance statistics based on permutation importance
- Rows: batch/feature combinations
- Columns: batch number (BATCH); feature names (FEAT); feature cluster: to avoid issues with multicollinear features during permutation importance, feature clustering is performed. Integer indicating which cluster the feature belonged to for that batch (-1 = no cluster) (CLUST); permutation importance score: mean change in super learner perfomance across rounds of permutation (features belonging to a cluster are always permuted simultaneously and the assigned score for each is scaled to the number of features in that cluster) (SCORE); probability of the score: probability of performance change relative to the best (optimized, non-permuted) score (P) 

## Analysis Summary:
The analysis pipeline consists of:

### Building and Training the Stacked Generalizer (Super Learner)
------
1) Multiple machine learning algorithms\* (submodels) are tuned using randomized parameter search with cross-validation and trained on the current batches' full training set. Submodels currently available include elastic net trained with stochastic gradient decsent (glm), non-linear support vector machine (svm), multilayer perceptron (mlp), random forest (randForest), and gradient boosting machine (gradBoost). All submodels aside from the random forest and gradient boosting machines are built in a bagging (bootstrap aggregation) ensemble to ensure comparable performance to random forest and gradient boosting. All submodels are optimized using mean absolute error.
2) Prediction submodels are aggregated into a super learner<sup>2,3</sup>, a linear combination of submodel predictions optimizing the same loss function, implemented as an elastic net model with no intercept and trained with stochastic gradient descent. The super learner is tuned using randomized paramter search, trained on the cross validation hold-outs from the full training sample, 

\*All algorithms are implemented using *SciPy*, *Statsmodels*, and *SciKit-Learn*, with many more to come.

### Evaluating the Super Learner and Submodel Performances
------
1) The normative super learner is used to make predictions in an 'atypical' sample displaying some phenotype-of-interest (eg. "symptomatic", "disease present"). Multiple atypical samples (phenotypes) can be predicted simultaneously, but should be labeled as separate groups in the CSV file (column 3)
2) Predictions for the atypical sample/s are used to calculate phenotype-specific deviations from the normative prediction (e.g. BrainAGE<sup>4</sup>)
3) Atypical sample predictions and deviations are output.

### Feature Importance Analysis using Permutation Importance
------
1) A univariate noise perturbation sensitivity (NPS)<sup>5</sup> analysis is used to interogate the magnititue and direction of feature influence on atypical deviations from the normative model. NPS is a feature-wise metric representing how sensitive group-level deviations from normative prediction are to the atypical phenotype.
2) Feature influence is thresholded using the paired-samples Wilcoxon test (perturbed deviation distribution vs. true deviation distribution) corrected for multiple comparisons using the Benjamini and Hochberg method. Features influencing non-significant differences in deviation distribution (based on the phenotype-of-interest OR by-chance during permutation testing) are considered non-influencial.
3) All feature influence scores, direction of influence (increasing or decreasing deviations from the norm), and associated descriptive statistics (distribution medians, Wilcoxon statistic, effect size) are output locally.

More detailed information to come! For specific questions, contact tjkeding@gmail.com.


## License
© Taylor J. Keding, 2020. Licensed under the General Public License v3.0 (GPLv3).
This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute (but please cite this repository when appropriate; contact tjkeding@gmail.com for more details).


## Acknowledgements & Contributions
Special thanks to Justin Russell Ph.D., Josh Cisler Ph.D., and Jerry Zhu Ph.D. (University of Wisconsin-Madison) for their input on implementation and best-practices. stacked-feature-learning is open for improvements and maintenance. Your help is valued to make the package better for everyone!


# References
**(1)** Marquand, A.F., Kia, S.M., Zabihi, M. et al. Conceptualizing mental disorders as deviations from normative functioning. *Molecular Psychiatry* 24, 1415–1424 (2019). DOI: 10.1038/s41380-019-0441-1

**(2)** van der Laan, M.J., Polley, E.C., & Hubbard, A.E. Super learner. *Statistical Applications in Genetics and Molecular Biology*. 6,25 (2007). DOI: 10.2202/1544-6115.1309

**(3)** Naimi, A.I. & Balzer, L.B. Stacked Generalization: An Introduction to Super Learning. *European Journal of Epidemiology*. 33, 459–464 (2018). DOI: 10.1007/s10654-018-0390-z

**(4)** Franke, K. & Gaser, C. Ten Years of BrainAGE as a Neuroimaging Biomarker of Brain Aging: What Insights Have We Gained? *Frontiers in Neurology*. 10, (2019). DOI: 10.3389/fneur.2019.00789

**(5)** Saltelli, A. Sensitivity analysis for importance assessment. *Risk Analysis*. 22, 3:579-90 (2002). DOI: 10.1111/0272-4332.00040
