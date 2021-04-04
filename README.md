# Udacity Machine Learning Engineer with Microsoft Azure - Capstone Project

This is the final project in the Udacity Azure Machine Learning Engineer Nanodegree. In this project, we created two models: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. We then compare the performance of both the models and deploy the best performing model.

Architectural Diagram![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/Architectural%20Diagram.png)

## Dataset
Name: heart_failure_clinical_records_dataset.csv

### Overview
I have downloaded the dataset from "UC Irvine Machine Learning Repository"

Heart failure (HF) occurs when the heart cannot pump enough blood to meet the needs of the body. Available electronic medical records of patients quantify symptoms, body features, and clinical laboratory test values, which can be used to perform biostatistics analysis aimed at highlighting patterns and correlations otherwise undetectable by medical doctors. Machine learning, in particular, can predict patients’ survival from their data and can individuate the most important features among those included in their medical records.

### Task
Task: This is a classification problem where in I'm trying to predict if the symptoms used in the features will cause death in the patient.(Yes or No)
The target variable is "death event".

Thirteen (13) clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- death event: if the patient deceased during the follow-up period (boolean)

## Automated ML

1. I created a Compute Instance with specification "STANDARD_D2_V2" to run Jupyter Notebook in Azure.
2. I have imported the dataset using TabularDataset library.
3. The setting that I used for Auto-ML were 
* "experiment_timeout_minutes", 
* "enable_early_stopping", 
* "n_cross_validations", 
* "max_concurrent_iterations"

automl_settings = {"primary_metric":"accuracy", "experiment_timeout_minutes":30, "enable_early_stopping":True, "n_cross_validations":3,"max_concurrent_iterations": 4}

4. automl_config = AutoMLConfig(compute_target = compute_target, task = 'classification', training_data = train, label_column_name = 'DEATH_EVENT',blocked_models=['XGBoostClassifier'],**automl_settings)

### Results

1. The best performing Algorithm was "VotingEnsemble" with an accuracy of 86.61%
![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/automl%20best%20model.png)

2. Run Details 
![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/automl%20run%20details.png)

## Hyperparameter Tuning

1. I have used LogisticRegression for this experiment .
2. I have used RandomParameterSampling with 3 parameters for this model:
solver
max_iter
C

RandomParameterSampling({'C': choice(0.01, 0.1, 1, 10, 100),
                                        'max_iter' : choice(50,75,100,125,150,175,200),
                                        'solver' : choice('liblinear','sag','lbfgs', 'saga')})

3. I have used the primary metric as "Accuracy" for this problem and I have tried to maximize it.


### Results

1. The best performing accuracy was 89.33% 
2. The parameters of the model are:
['--C', '0.1', '--max_iter', '50', '--solver', 'liblinear']
3. I could increase the number of parameter ranges that I have used.
I can even change the method of sampling used for the execution to run faster or slower and find good accurate results.

Best Model :
![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/hyperdrive%20best%20model.png)

Run Details:
![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/hyperdrive%20run%20details.png)

## Model Deployment

Since the HyperDrive experiment gave me best metrics i.e Accuracy of 89.33%, I deployed this model. 

Deployed Service :
![](https://github.com/aishpn5/Udacity-Azure-ML-Capstone-project/blob/main/Images/endpoint%20active.png)

## Screen Recording
Link: https://youtu.be/7vKovlEx6jc
