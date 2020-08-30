# CreditRisk

## A. Background 

Evaluate Lending Club data using various machine model techniques namely 
oversampling (randomoversampler), SMOTE Oversampling, undersampling (centroid),
over/under sampling (SMOTEENN), BalancedRandomForestClassifier, EasyEnsemblrClasifer

## B. Method
1. The non-numeric columns were converted using LabelEncoder, and by applying pandas 
conversion with a lamda function (e.g loan_status : low_risk =0, high_risk=1)

2. Dropped some of the columns that probably have little impact to the outcome: next_pymnt_d,issue_d,

3. The dataframe of interest is called loan_df

The data was split standard : 75%,25%

## C. Results

1. **RandomOverSampler**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		9971	7133
		high_risk(1)		32	69

	b. Balanced Accuracy Score : 0.63	

	c. Report

                   		pre       rec       spe        f1      
          low_risk(0)       	1.00      0.58      0.68      0.74    
          high_risk(1)       	0.01      0.68      0.58      0.02


        d. For high risk, this model has low  precision (0.01), and  moderate 
	sensitivity of 0.68.  This is also indicated by the F1 value (0.02). 
	Accuracy at moderate level of 0.63 

2. **SMOTE**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		11486	5618
		high_risk(1)		43	58

	b. Balanced Accuracy Score : 0.62

	c. Report


                   		pre       rec       spe        f1       
          low_risk(0)       	1.00      0.67      0.57      0.80      
          high_risk(1)       	0.01      0.57      0.67      0.02   


	d. For high_risk, this model has low  precision (0.01), and moderate 
	sensitivity of 0.57.  This is also indicated by the F1 value (0.02).
	RandomOverSampler slightly better in sensitivity. Accuracy at moderate 
	level of 0.62

3. **UNDERSAMPLING : ClusterCentroids**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		7145	9959
		high_risk(1)		34	67

	b. Balanced Accuracy Score : 0.54

	c. Report

                   		pre       rec       spe        f1        
          low_risk(0)       	1.00      0.42      0.66      0.59       
          high_risk(1)       	0.01      0.66      0.42      0.01       

	d. For high_risk, this model has low  precision (0.01), and moderate 
	sensitivity of 0.66.  This is also indicated by the F1 value (0.01).
	Not much better. For low risk, sensitivity goes down.
	Accuracy is also down to 0.54 

4. **UNDERSAMPLING : SMOTEENN**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		9591	7513
		high_risk(1)		24	77

	b. Balanced Accuracy Score : 0.66

	c. Report

                   		pre       rec       spe        f1       
          low_risk(0)       	1.00      0.56      0.76      0.72      
          high_risk(1)       	0.01      0.76      0.56      0.02            

	d. For high_risk, this model has low  precision (0.01), and moderate/good 
	sensitivity of 0.76.  This is also indicated by the F1 value (0.02).
	Slightly better on sensitivity for high risk, at the expense of low_risk 
	sensitivity.Accuracy at moderate level of 0.66, slightly better than others.

5. **BalancedRandomForestClassifier**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		14220	2884
		high_rish(1)		42	59

	b. Balanced Accuracy Score : 0.71

	c. Report

                   		pre       rec       spe        f1        
          low_risk(0)       	1.00      0.83      0.58      0.91       
          high_risk(1)       	0.02      0.58      0.83      0.04                 

	d. For high_risk, this model has low  precision (0.02), and moderate
	sensitivity of 0.58.  This is also indicated by the F1 value (0.04).
	Slightly better on sensitivity for low risk, at the expense of high risk 
	sensitivity. Accuracy is better at 0.71 with this model, ~10% improvement.

6. **EasyEnsembleClassifier**

	a. Confusion Matrix

				Predicted 0	Predicted 1
		low_risk(0)		13129	3975
		high_risk(1)		29	72

	b. Balanced Accuracy Score : 0.74

	c. Report

                   		pre       rec       spe        f1       
          low_risk(0)       	1.00      0.77      0.71      0.87       
          high_risk(1)       	0.02      0.71      0.77      0.03                       

	d. For high_risk, this model has low  precision (0.02), and good
	sensitivity of 0.71.  This is also indicated by the F1 value (0.03).
	Slightly better on sensitivity for low risk and high risk.
	Accuracy is better at 0.74 with this model. Ensemble method provides
	better accuracy with this data set.     

## D.Recommendation

	Seems the better method is **EasyEnsemblerClassifier**, with 100 estimators.
	This provides a better accuracy  at ~0.75 level and sensitivity for both 
	low_risk and high_risk around ~0.77. Low_risk precision is good in all 
	cases at 1.00, but high_risk precision needs much more work improvement, 
	as most models have a lot of false negatives and fair amount of 
	**false positives**. False positives are a problem as the key prediction for 
	this problem is missed.	**Undersampling:SMOTEN** has the best for 
	**false positives** at 24, slightly better than EasyEnsemblerClassifier. 
	Results are as good as the data sample avaialbe and the input parameters. 
	In our model we used all possible numeric columns for the input set. As part
	of the BalancedRandomForestClassifier, we looked at descending order of 
	probability for	the various field , the top ten list was :
	[(0.08656635716268982, 'last_pymnt_amnt'),
 	(0.06712830386036822, 'total_rec_prncp'),
 	(0.06455572411377343, 'total_rec_int'),
 	(0.057853975663494615, 'total_pymnt_inv'),
 	(0.048210348259077704, 'total_pymnt'),
 	(0.03736431527713882, 'int_rate'),
 	(0.02006311966510871, 'dti'),
 	(0.018243744205644088, 'max_bal_bc'),
 	(0.01777158070527311, 'mths_since_rcnt_il'),
 	(0.01764144706176067, 'installment')]
	We can re-run the model with these columns, we dont expect much improvement, 
	but there may be marginal improvement.
