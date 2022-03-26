# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
pd.options.display.float_format = '{:.2f}'.format
import time

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
#from PIL import Image

# loading the trained model

pickle_in = open(pathlib.Path.cwd().joinpath('project_app','model_regressor.pkl'), 'rb') 
model_regressor = pickle.load(pickle_in)

pickle_in2 = open(pathlib.Path.cwd().joinpath('project_app','scaler.sav'), 'rb') 
scaler_regressor = pickle.load(pickle_in2)


pickle_in3 = open(pathlib.Path.cwd().joinpath('project_app','model_regressor_uq.pkl'), 'rb') 
model_regressor_upper = pickle.load(pickle_in3)


pickle_in4 = open(pathlib.Path.cwd().joinpath('project_app','model_regressor_lq.pkl'), 'rb') 
model_regressor_lower = pickle.load(pickle_in4)
@st.cache(suppress_st_warning=True) 


def get_df(room_type_option,neighborhood,  minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365, amenities_500, leisure_500, subway_500, natural_500):
 


    # Pre-processing user input    
    if (room_type_option=='Shared Room'):
        room_type=0
    elif (room_type_option=='Private Room'):
        room_type=1
    else:
        room_type=2

    if (neighborhood=='Manhattan'):
          NG_Brooklyn=0
          NG_Manhattan=1
          NG_Queens=0
          NG_Staten_Island=0
      
    elif (neighborhood=='Brooklyn'):
          NG_Brooklyn=1
          NG_Manhattan=0
          NG_Queens=0
          NG_Staten_Island=0
          
    elif (neighborhood=='Queens'):
        NG_Brooklyn=0
        NG_Manhattan=0
        NG_Queens=1
        NG_Staten_Island=0
        
    elif neighborhood=='Staten Island':
        NG_Brooklyn=0
        NG_Manhattan=0
        NG_Queens=0
        NG_Staten_Island=1
    else:
        NG_Brooklyn=0
        NG_Manhattan=0
        NG_Queens=0
        NG_Staten_Island=0

    user_report_data = {
      'room_type':room_type,
      'minimum_nights':minimum_nights,
      'number_of_reviews':number_of_reviews,
      'reviews_per_month':reviews_per_month,
      'calculated_host_listings_count':calculated_host_listings_count,
      'availability_365':availability_365,
      'NG_Brooklyn':NG_Brooklyn,
      'NG_Manhattan':NG_Manhattan,
      'NG_Queens':NG_Queens,
      'NG_Staten_Island':NG_Staten_Island,
      #'latitude':latitude,
      #'longitude':longitude,
      'amenities_500':amenities_500,
      'leisure_500':leisure_500,
      'subway_500':subway_500,
      'natural_500':natural_500
      
      }
    df_report = pd.DataFrame(user_report_data,index=[0])
    

    
    df_std = pd.DataFrame(scaler_regressor.transform(df_report),
                                            index=df_report.index,
                                            columns=df_report.columns)
    
    return df_std

def prediction(df):
    pred = model_regressor.predict(df)
    pred_upper = model_regressor_upper.predict(df)
    pred_lower = model_regressor_lower.predict(df)
    #pred = np.expm1(salary)
    
    return pred, pred_upper, pred_lower



def main():
    

    html_temp = """ 
    <div style ="background-color:gray;padding:13px"> 
    <h1 style ="color:black;text-align:center;">CityWise AI App</h1> 
    </div> 
    """

    image = Image.open(pathlib.Path.cwd().joinpath('project_app','Webapp_image.png'))
    
    


    


    st.image(image, 'CityWise AI', width = 200)
    st.header("Vacation Rentals Price Predictor")

    st.markdown("Select the options on the sidebar to determine the price of your Airbnb listing")
    
    
    
    
    
    

    # display the front end aspect
    
    #st.markdown(html_temp, unsafe_allow_html = True) 
    

    #city = st.sidebar.text_input("City", "Brooklyn")
    country = st.sidebar.selectbox("Country", ["United States"])
    state = st.sidebar.selectbox("State", ["New York"])
    neighborhood = st.sidebar.selectbox('New York Neighbourhood',
                ('Brooklyn', 'Manhattan', 'Queens','Bronx','Staten Island'))
    street = st.sidebar.text_input(" Street", "341 Eastern Pkwy")   
    tax_rate = st.sidebar.slider('Tax Rate', 0.0,1.0, 0.01 )



    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street+", "+neighborhood+", "+state+" ,"+country)


    try:
        latitude = location.latitude
        longitude = location.longitude
    except:
        st.error('There is an error with your location. Please check.')
        e = RuntimeError('This is an exception of type RuntimeError')
        st.exception(e)
        exit ()

    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

    st.map(map_data, zoom=15, use_container_width=True) 
  

    
    room_type_option = st.sidebar.selectbox('Room Type',('Shared Room','Private Room','Entire house'))
    minimum_nights = st.sidebar.text_input('Minimum Nights', 0,30, 1 )
    number_of_reviews = st.sidebar.text_input('Number of reviews', 0,629, 1 )
    reviews_per_month = st.sidebar.text_input('Reviews per month', 0,58, 1 )
    calculated_host_listings_count = st.sidebar.text_input('Number of host listings', 1,327, 1 )
    availability_365= st.sidebar.text_input('Availability', 0,365, 1)
    sideb = st.sidebar

    # when 'Predict' is clicked, make the prediction and store it 
    if sideb.button("Predict Price"): 
        ox.config(log_console=True, use_cache=True)

        tag_leisure = {
                'leisure':['park','water-park','amusement-park','theme-park','zoo','stadium']}
        tag_subway = {'building': ['train_station']}
        tag_natural = {'natural': ['beach', 'park', 'water']}
        tag_amenities = {'amenity': ['restaurant', 'pub', 'hotel'],
                'building': ['hotel','transportation','airport'],'store':'mall',
                'tourism': 'hotel'}
        
        

        
        with st.spinner('Calculating the number of amenities in a 500m radio...'):
            time.sleep(10)
        try:
            gdf_amenities = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_amenities) # Boundary to search within
        except:
            gdf_amenities = 10
        with st.spinner('Calculating the number of leisure locations in a 500m radio...'):
            time.sleep(10)
        try:
            gdf_leisure = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_leisure) # Boundary to search within
        except:
            gdf_leisure = 10
        with st.spinner('Calculating the number of train stations in a 500m radio...'):
            time.sleep(10)
        try: 
            gdf_subway= ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_subway) # Boundary to search within
        except:
            gdf_subway = 10
        with st.spinner('Calculating the number of natural resources in a 500m radio...'):
            time.sleep(10)
        try:
            gdf_natural = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_natural) # Boundary to search within
        except:
            gdf_natural = 10
            
        st.success('Done')
        st.write('Country Selected:', country)
        st.write('State Selected:', state)
        st.write('Neighborhood Selected:', neighborhood)
        st.write('Street Selected:', street)
        
        amenities_500 = gdf_amenities.shape[0]
        st.write(f'Number of restaurants, airports, malls, hotels or pubs: {amenities_500}')
        subway_500 = gdf_subway.shape[0]
        st.write(f'Number of Train Stations: {subway_500}')
        natural_500 = gdf_natural.shape[0]
        st.write(f'Number of Baches and Parks:{natural_500}')
        leisure_500 = gdf_leisure.shape[0]
        st.write(f'Number of  zoos, theme-parks, water-parks, and stadiums: {leisure_500}')
        user_data = get_df(room_type_option,neighborhood, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365, amenities_500, leisure_500, subway_500, natural_500)
        
        pred, pred_upper, pred_lower = prediction(user_data)
        st.write('Price prediction per night')

        
        st.balloons()
        
        annual_revenue=np.round(pred[0]*120,2)
        calculate_tax=np.round(pred[0]*120*tax_rate,2)
        col1, col2  = st.columns(2)
        st.write(f"Price Estimate per night: $ {str(np.round(pred[0], 2)}")
        st.write(f'The acceptable range: ${str(np.round(pred_lower[0], 2))} -  ${str(np.round(pred_upper[0], 2))}')
        col1.metric("Revenue: $",str(annual_revenue))
        col2.metric("Annual Assessed Tax: $", str(calculate_tax))
        st.caption("Revenue and tax is calculated assuming that unit is rented for 120 days in a year ")
        #st.subheader(f'Assessed taxes: ${str(np.round(pred[0]*120*tax_rate, 2))}')
        
    
if __name__=='__main__': 
    main()


