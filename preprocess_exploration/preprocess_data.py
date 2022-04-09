#import libraries
import numpy as np
import pandas as pd
#Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#scipy
from scipy.stats.mstats import winsorize
#utils functions
from utils_func import scaler_transform

def preprocess(dataset,geodata = 'False'):
    df = dataset.copy(deep = True)
    #drop duplicates
    df.drop_duplicates(inplace=True)
    #data types
    for _ in ['id','host_id']:
        df[_] = df[_].astype('object')
    df_missing = df.copy(deep=True)
    #dropping irrelevant columns
    df_missing.drop(['host_id','host_name'], axis=1, inplace=True)
    #replacing missing values
    df_missing[(df_missing.last_review.isnull()) & (df_missing.reviews_per_month.isnull())] = df_missing[(df_missing.last_review.isnull()) & (df_missing.reviews_per_month.isnull())].replace(np.nan,0)
    #dropping irrelevant columns
    df_missing.drop(['last_review'], axis=1, inplace=True)
    #dropping rows with NaN in listing names
    df_missing.dropna(subset = ['name'], inplace= True)
    df_missing['name'] = df_missing.name.astype('str')


    #feature encoding
    df_encode = df_missing.copy(deep=True)
    df_encode = pd.get_dummies(df_encode, columns = ['neighbourhood_group'],
                                          prefix = 'NG',drop_first=True)
    df_encode.drop(['neighbourhood'], axis=1, inplace=True)
    df_encode['room_type']=pd.factorize(df_encode.room_type)[0]

    #truncate minimum nights
    from scipy.stats.mstats import winsorize
    df_win = df_encode.copy(deep=True)
    df_win['minimum_nights'] = winsorize(df_win['minimum_nights'], limits=(0, 0.05))

    #to remove the skeweness of the price
    df_log = df_win.copy(deep=True)
    df_log['price'] =  np.log1p(df_log['price'])
    df_all = df_log.copy(deep=True)
    if geodata == 'True':
        #integrate the geodata dataframe
        geodata = pd.read_csv('../merged_map.csv')
        df_all = df_all.merge(geodata, left_on = 'id', right_on = 'id')
        df_all.drop(['geom' ], axis=1, inplace=True)
    #scale the numeric features 
    df_std = df_all.copy(deep=True)
    df_std, scaler = scaler_transform('standard', df_std, exclude_vars = ['price','latitude', 'longitude', 'id','NG_Brooklyn','NG_Manhattan','NG_Queens','NG_Bronx','NG_Staten Island','room_type_Private room','room_type_Shared room'])
    for _ in ['id','name']:
        df_std[_] = df_std[_].astype(str)
    preprocessed_df = df_std.copy(deep=True)

    return preprocessed_df