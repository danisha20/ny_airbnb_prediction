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
files_dir = "../dataset/"
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

tag_leisure = {
        'leisure':['park','water-park','amusement-park','theme-park','zoo','stadium']}
tag_subway = {'building': ['train_station']}
tag_natural = {'natural': ['beach', 'park', 'water']}
tag_amenities = {'amenity': ['restaurant', 'pub', 'hotel'],
        'building': ['hotel','transportation','airport'],'store':'mall',
        'tourism': 'hotel'}

distance = 500

#%%
#map libraries
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import geopy
import osmnx as ox

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image
import pathlib
street = "341 Eastern Pkwy"
neighborhood = 'Brooklyn'
state = "New York"
country = "United States"
geolocator = Nominatim(user_agent="GTA Lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geolocator.geocode(street+", "+neighborhood+", "+state+" ,"+country)

#%%
latitude = location.latitude
longitude = location.longitude
#%%
gdf_amenities = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_amenities) # Boundary to search within
gdf_leisure = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_leisure)
gdf_subway= ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_subway)
gdf_natural = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_natural) 
#%%
gdf_amenities = gdf_amenities.loc[:,['amenity','geometry']]
gdf_leisure = gdf_leisure.loc[:,['leisure','geometry']]
gdf_subway = gdf_subway.loc[:,['building','geometry']]
gdf_natural = gdf_natural.loc[:,['natural','geometry']]


#%%
gdf_natural
#%%
gdf_amenities['Center_point'] = gdf_amenities['geometry'].centroid
gdf_leisure['Center_point'] = gdf_leisure['geometry'].centroid
gdf_subway['Center_point'] = gdf_subway['geometry'].centroid
gdf_natural['Center_point'] = gdf_natural['geometry'].centroid
#%%
gdf_amenities["long"] = gdf_amenities.Center_point.map(lambda p: p.x)
gdf_amenities["lat"] = gdf_amenities.Center_point.map(lambda p: p.y)
gdf_leisure["long"] = gdf_leisure.Center_point.map(lambda p: p.x)
gdf_leisure["lat"] = gdf_leisure.Center_point.map(lambda p: p.y)
gdf_subway["long"] = gdf_subway.Center_point.map(lambda p: p.x)
gdf_subway["lat"] = gdf_subway.Center_point.map(lambda p: p.y)
gdf_natural["long"] = gdf_natural.Center_point.map(lambda p: p.x)
gdf_natural["lat"] = gdf_natural.Center_point.map(lambda p: p.y)

#%%
gdf_amenities['type'] = 'amenities'
gdf_leisure['type'] = 'leisure'
gdf_subway['type'] = 'subway'
gdf_natural['type'] = 'natural'
#%%
all_geom = pd.concat([gdf_amenities,gdf_leisure,gdf_subway,gdf_natural], ignore_index=True)
all_geom_count = all_geom.groupby('type')['geometry'].count()
all_geom_count = pd.DataFrame(all_geom_count).reset_index()
#%%

import plotly.express as px
fig = px.bar(all_geom_count, x='type', y='geometry')
fig.show()

#%%
all_geom
#%%
import geopandas
gdf = geopandas.GeoDataFrame(all_geom, geometry='geometry')

#%%
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#%%
ax = world[(world.continent == 'North America') & (world.name == 'United States of America')].plot(
    color='white', edgecolor='black')

gdf.plot(ax=ax, color='red')

plt.show()

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
