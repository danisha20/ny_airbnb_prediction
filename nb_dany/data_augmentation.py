#%%
#import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly_express as px
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import osmnx as ox
import seaborn as sns
#importing dataset
files_dir = "dataset/"
filename = files_dir + "clean_dataset_airbnb.csv"
df = pd.read_csv(filename)
df.head()

#%%
#visualize
# token2 = "pk.eyJ1IjoiZGFuaXNoYTIwIiwiYSI6ImNrenJ0Z2E0YjJreHUydnFvbzBpbGIwMWoifQ.bUkA0cPk8lgAmvR1ePc0yg"
# px.set_mapbox_access_token(token2)
# fig = px.scatter_mapbox(df,
#                         lat="latitude",
#                         lon="longitude",
#                         hover_name="id",
#                         zoom=10)
# fig.show()

#%%
df["geom"] =  df["latitude"].map(str)  + ',' + df['longitude'].map(str)
#%%
df['amenities_500'] = None
df['leisure_500'] = None

#%%
ox.config(log_console=True, use_cache=True)

tag_amenities = {'amenity': ['restaurant', 'pub', 'hotel'],
        'building': ['hotel','transportation','airport'],'store':'mall',
        'tourism': 'hotel'}
tag_leisure = {
        'leisure':['park','water-park','amusement-park','theme-park','zoo','stadium']}
distance = 500
# %%
batch_list = [0,5,10,15]
for batch in range(len(batch_list)-1):
        i = batch_list[batch]
        j = batch_list[batch+1]
        for element, row in df.iloc[i:j,:].iterrows():
                ox.config(log_console=True, use_cache=True)
                gdf_amenities = ox.geometries.geometries_from_point((df.iloc[element,:]['latitude'],df.iloc[element,:]['longitude']),dist = distance, tags = tag_amenities) # Boundary to search within
                df.loc[element, 'amenities_500']  = gdf_amenities.shape[0]
                gdf_leisure = ox.geometries.geometries_from_point((df.iloc[element,:]['latitude'],df.iloc[element,:]['longitude']),dist = distance, tags = tag_leisure) # Boundary to search within																			dist=distance, tags=tags)
                df.loc[element, 'leisure_500']  = gdf_leisure.shape[0]
        subset_data = df[['id','geom', 'amenities_500', 'leisure_500']].iloc[i:j,:]
        subset_data.to_csv(f'collected_map/batch-{i}-{j}.csv', index=False)
        
# %%
df[['id','amenities_500','leisure_500']].head(25)
# %%
batch_list =np.arange(start = 0, stop = df.shape[0], 
          step = 500)
# %%
batch_list
# %%
df
# %%
