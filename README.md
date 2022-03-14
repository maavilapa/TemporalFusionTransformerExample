<h1 align='center'>Time series forecasting with Temporal fusion transformer (TFT)</h1>

<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/images/fig_8.PNG" width=800>
</p>

## Business case
## Table of contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Summary ](#Summary)
   * [ 1. EDA and Cleaning ](#EDA_and_Cleaning)
       * [ Imports and data download](#Imports_and_data_download)
       * [ Checking data types and missing values](#Checking_data_types_and_missing_values)
       * [ Filling missing values](#Filling_missing_values)
   * [ 2. Preprocessing ](#Preprocessing) 
       * [ Scaling](#Scaling)
       * [ Create test dataframe](#Split_data)
       * [ Date features](#Date_features)       
   * [ 3. Training ](#Training)
       * [ Training parameters](#Training_parameters)
       * [ Create datasets](#Create_datasets)
       * [ Hyperparameter tuning](#Hyperparameter_tuning)  
       * [ Predictions on validation data](#Predictions_on_validation_data)
       * [ Training and validation plots](#Training_and_validation_plots)
       * [ Predict on test data](#Predict_on_test_data)
  * [ 4. Future improvements](#Future_improvements)

</details>

## File descriptions

<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ data ](https://github.com/maavilapa/TemporalFusionTransformerExample/data)</strong>: folder containing all data files
    * <strong>sample_submission.csv</strong>: a sample submission file in the correct format
    * <strong>store.csv</strong>:  supplemental information about the stores
    * <strong>test.csv</strong>:  historical data excluding Sales
    * <strong>train.csv</strong>:  historical data including Sales

* <strong>[ images ](https://github.com/maavilapa/TemporalFusionTransformerExample/images)</strong>: folder containing images used for README and preparation notebook
* <strong>[ preparation](https://github.com/maavilapa/TemporalFusionTransformerExample/preparation)</strong>: Functions used in the data preparation notebook used for preprocessing and training the model. 
* <strong>[ 1-2. Data preparation](https://github.com/maavilapa/TemporalFusionTransformerExample/1-2._Data_preparation.ipynb)</strong>: Notebook with all the data preparation, model training and predictions process.
</details>



## Technologies used
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>tensorflow</strong>
* <strong>tensorboard</strong>
* <strong>pytorch_lightning</strong>
* <strong>Scikit-Learn</strong>
* <strong>pytorch_forecasting</strong>

</details>
## Summary

### EDA and Cleaning

#### Imports and data download

#### Checking data types and missing values

#### Filling missing values

### Preprocessing 

#### Scaling

#### Create test dataframe

#### Date features

### Training

#### Training parameters

#### Create datasets

#### Hyperparameter tuning

#### Predictions on validation data

#### Training and validation plots

#### Predict on test data

This is my project about time series forecasting
