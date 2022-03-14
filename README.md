<h1 align='center'>Time series forecasting with Temporal Fusion Transformer (TFT) in Rossman store sales dataset</h1>

<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_8.PNG" width=500>
</p>

## Business case
Most advanced methods applied on time series include deep learning models, specially for complex multivariate forecasting problems. One of them, the Temporal fusion
transformers (TFT), has demonstrated significant performance improvements over existing benchmarks and is currently one of the most accurate methods in forecasting. Although it is an advanced model, its implementation in the pytorch_forecasting library is user friendly and provides all the necessary functions to plot and interpret the model predictions. Besides, if it is combined with tools like tensorboard, tensorflow data validation and featurewiz, we can create a pipeline to prepare, add features and predict data in time series datasets in a flexible and understandable way. 

I will show this process using the Rossman store sales dataset, one of the open timeseries datasets available in Kaggle. We are provided with historical sales data for 1,115 Rossmann stores, using not just the sales historical of each store, but information about promotions, number of clients and holidays. Some stores in the dataset were temporarily closed for refurbishment and therefore, we have to clean the data and fill the missing values. The data frequency is daily and we have to predict the "Sales" for some of the stores given in the test set in the next 48 days. Sellers could take advantage of this kind of predictions during their inventory planning, particularly when a lot of data about products sales and promotions in each store is available. 
## Table of contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Summary ](#Summary)
   * [ 1. EDA and Cleaning ](#EDA_and_Cleaning)
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

<details open>
<summary>Show/Hide</summary>
<br>

<a name="Import and data download"></a>

#### Checking data types and missing values
The first step is to import the necessary libraries defined in the requirements.txt file and to download the train.csv and store.csv files as pandas dataframes. Using tensorflow data validation we could check the data type for each column in both Train and Store dataframes, the percentage of zeros, missing values and more statistical information of each column.
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_1.PNG" width=600>
</p>
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_2.PNG" width=600>
</p>
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_3.PNG" width=600>
</p>
For categorical features we can check not just the missing values, but also the Top value and the unique values.
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_4.PNG" width=600>
</p>
We found that Promo2SinceYear, Promo2SinceWeek, CompetitionOpenSinceMonth, CompetitionOpenSinceYear and PromoInterval have more than 30% of missing values. CompetitionDistance has 0.26% of missing values. We will drop the first two columns and fill the missing values in the CompetitionDistance column to use it as a training feature. We merge the Train and Store dataframes. We check the number of missing values by columns using the next line: 
  
```bash
train_data.isnull().sum()
```
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_4a.PNG" width=300>
</p>

We check that there are 2642 missing values of the 1017209 rows and records. If we look at which are the stores with missing values we find the following 3 stores: <strong>[291 622 879]</strong>.
  
#### Filling missing values
We first fill the CompetitionDistance Null values using the median Competition distance of the other 1112 stores. To do that we run the next line:
  
```bash
train_data.CompetitionDistance=train_data.CompetitionDistance.fillna(train_data.CompetitionDistance.median())
```
It is necessary to check if a store has more than one Sales record for each unique date and if each date has a sales value for each store, since by default pytorch forecasting timeseries datasets allow missing timesteps but fill them with 0. We use the pandas duplicated, group by and value_counts functions to check the number of days recorded by store. 

<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig4_b.PNG" width=300>
</p>

We found that from 1115 stores, 934 have sales recorded for all the 942 days considered in the train data, while 180 Stores have just 758 sales recorded and 1 has 941 days. We filter the stores that have missing sales values and can plot the historical sales for some of these stores.

```bash
train_data[["Date", "Store", "Sales"]][train_data.Store==539].set_index("Date").plot()
```
<p align="center">
  <img src="https://github.com/maavilapa/TemporalFusionTransformerExample/blob/main/images/fig_4c.PNG" width=400>
</p>
  
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
