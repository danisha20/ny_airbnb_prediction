#import libraries
import numpy as np
import pandas as pd
#Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#importing dataset
files_dir = "../dataset/"
filename = files_dir + "AB_NYC_2019.csv"
df = pd.read_csv(filename)

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

#feature encoding
df_dummies = pd.get_dummies(df_missing, columns = ['neighbourhood_group'],
                                          prefix = 'NG',drop_first=True)

df_dummies = pd.get_dummies(df_dummies, columns = ['room_type'],
                                          drop_first=True)
df_dummies.drop(['neighbourhood'], axis=1, inplace=True)

#handling outliers
# from scipy.stats.mstats import winsorize
# df_win = df.copy(deep=True)
# df_win['minimum_nights'] = winsorize(df['minimum_nights'], limits=(0, 0.075))
# df_win['number_of_reviews'] = winsorize(df['number_of_reviews'], limits=(0, 0.075))
# df_win['reviews_per_month'] = winsorize(df['reviews_per_month'], limits=(0, 0.075))
# df_win['calculated_host_listings'] = winsorize(df['calculated_host_listings_count'], limits=(0, 0.075))
# df_win['availability_365'] = winsorize(df['availability_365'], limits=(0, 0.075))


# #feature scaling
def scaler_transform(scaler_type, X, exclude_vars = ['latitude', 'longitude', 'id','NG_Brooklyn','NG_Manhattan','NG_Queens','NG_Bronx','NG_Staten Island','room_type_Private room','room_type_Shared room']):
    '''
    Transform the NON-object type data to the selected scaler
    '''
    X_copy = X.copy(deep = True)
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    X_notexcluded= X_copy.loc[:, ~X_copy.columns.isin(exclude_vars)]
    X_withoutobj = X_notexcluded.select_dtypes(exclude=['object'])
    X_withoutobj = X_withoutobj.add_suffix(f'_{scaler_type}')
    # Fit transform the scaler if there are objects in dataset
    if X_withoutobj.shape[1] > 0:
        X_withoutobj = pd.DataFrame(scaler.fit_transform(X_withoutobj[X_withoutobj.columns]),
                                        index=X_withoutobj.index,
                                        columns=X_withoutobj.columns)
        # Concatenate the rest of the data
        X_withexcl = pd.concat([X_withoutobj, X.loc[:, X.columns.isin(exclude_vars)]], axis=1)
        others = [x for x in X.select_dtypes('object') if x not in X_withexcl.columns]
        X_final = pd.concat([X_withexcl,X[others]], axis=1)
    return X_final
  
df_std = scaler_transform('standard', df_dummies)

# #export pre-processed dataset
# df_clean.to_csv('dataset/clean_dataset_airbnb.csv')