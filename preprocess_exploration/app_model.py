#%%

#import streamlit as st , use version 0.75
import pandas as pd

#import libraries
import numpy as np
import pandas as pd

from preprocess_data import preprocess

#importing dataset
files_dir = "../dataset/"
filename = files_dir + "AB_NYC_2019.csv"
df = pd.read_csv(filename)
preprocessed_df = preprocess(df)

preprocessed_df
