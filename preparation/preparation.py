
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pytorch_forecasting.data import  EncoderNormalizer
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import io as io2
import matplotlib.pyplot as plt
import tensorflow as tf

def reindexing(df, date_index):
    df_reindexed=pd.DataFrame()
    for store in df["Store"].unique():
      #print(itemloc)
      df_aux=pd.DataFrame()
      df_aux=df[df["Store"]==store]
      df_aux=df_aux.set_index("Date").reindex(date_index).reset_index()
      df_aux[["Store", "CompetitionDistance", "StoreType", "Assortment"]]=df_aux[["Store", "CompetitionDistance", "StoreType", "Assortment"]].ffill().bfill()
      df_reindexed=df_reindexed.append(df_aux, ignore_index=True)
    return df_reindexed

def add_last_year_values(df):
    df_reindexed=pd.DataFrame()
    for store in df["Store"].unique():
      #print(itemloc)
      df_aux=pd.DataFrame()
      df_aux=df[df["Store"]==store]
      df_aux[["previous_sales","previous_day","previous_customers", "previous_open", "previous_promo" ,"previous_state_holiday", "previous_school_holiday" ]]=df_aux.groupby([df_aux['Date'].dt.month,df_aux['Date'].dt.day])[['Sales', "DayOfWeek", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].shift().shift(-1)
      df_reindexed=df_reindexed.append(df_aux, ignore_index=True)
    return df_reindexed

def min_max_scaler(df, columns=["Customers","Sales"]):
        """
        Scales the data with a Min Max scaler.
        
        :param df: Input dataframe used to train the models predictions.
 
        :return scalers: Array with the scalers for each feature.
        :return data_train: Normalized input dataframe.

        """
        for column in columns:
          df_out=df[["Date", "Store", column]]
          df_out = df_out.sort_values(by='Date')
          df_out = df_out.reset_index()
          df_out = df_out.iloc[:, 1:]
          df_out = df_out.pivot(index='Date', columns='Store', values=column).reset_index()
          #df_out = df_out.fillna(value=value, method=method)
          df_out=df_out.set_index("Date")
          df_out =df_out.reindex(sorted(df_out.columns), axis=1)
          scalers={}
          for j in df_out.columns:
                  scaler = MinMaxScaler(feature_range=(-1,1))
                  s_s = scaler.fit_transform(df_out[j].values.reshape(-1,1))
                  s_s=np.reshape(s_s,len(s_s))
                  scalers['scaler_'+ str(j)] = scaler
                  df_out[j]=s_s
          df_out= df_out.stack()\
                  .reset_index(drop=False)\
                  .rename(columns={'level_1': 'Store', 0: column})\
                  .sort_values(by='Date',ascending=True)
          df=df.drop(columns=column).merge(df_out, on=["Date", "Store"])
        return scalers, df

def create_training_dataset(data,target, training_cutoff, group_ids, input_window,forecast_horizon, unknown_reals, known_reals, static_reals, unknown_categorical, known_categorical, static_categorical ):
        target_normalizer=EncoderNormalizer()
        training = TimeSeriesDataSet(
            data[data.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=target,
            group_ids=group_ids,
            min_encoder_length=forecast_horizon ,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=input_window,
            min_prediction_length=forecast_horizon,
            max_prediction_length=forecast_horizon,
            time_varying_unknown_reals=unknown_reals,
            time_varying_known_reals=known_reals,
            static_reals=static_reals,
            static_categoricals=static_categorical,
            time_varying_known_categoricals= known_categorical,
            time_varying_unknown_categoricals= unknown_categorical,
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,

        )
        return training

def rescale(scalers, df):
        p1=pd.DataFrame()
        for key in scalers.keys():
            aux=pd.DataFrame()
            scaler=scalers[key]
            aux=pd.DataFrame(scaler.inverse_transform(df[df.Store==key.replace("scaler_", "")].iloc[:,:7]), columns=["p5",	"p20",	"p40",	"p50",	"p60",	"p80",	"p95"])
            p1=p1.append(aux, ignore_index=True)
        p1["Store"]=df["Store"]
        p1["time_idx"]=df["time_idx"]
        return p1


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io2.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image