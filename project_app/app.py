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
          tax_rate=0.12
      
    elif (neighborhood=='Brooklyn'):
          NG_Brooklyn=1
          NG_Manhattan=0
          NG_Queens=0
          NG_Staten_Island=0
          tax_rate=0.14
          
    elif (neighborhood=='Queens'):
        NG_Brooklyn=0
        NG_Manhattan=0
        NG_Queens=1
        NG_Staten_Island=0
        tax_rate=0.11
        
    elif neighborhood=='Staten Island':
        NG_Brooklyn=0
        NG_Manhattan=0
        NG_Queens=0
        NG_Staten_Island=1
        tax_rate=0.10
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
    
    return df_std, tax_rate

def prediction(df):
    pred = model_regressor.predict(df)
    pred_upper = model_regressor_upper.predict(df)
    pred_lower = model_regressor_lower.predict(df)
    #pred = np.expm1(salary)
    
    return pred, pred_upper, pred_lower



def main():
    



    image = Image.open(pathlib.Path.cwd().joinpath('project_app','city.png'))
    

    st.image(image, width = 200)
    st.header('CityWise AI')
    st.subheader("A modern tool for urban planning")
    #st.subheader("Vacation Rentals Price Predictor")

    
    
    st.sidebar.header('Select criteria')
    st.sidebar.markdown("Select the options to determine the price of your listing.")
    
    


    #city = st.sidebar.text_input("City", "Brooklyn")
    street = st.sidebar.text_input(" Street", "341 Eastern Pkwy")
    neighborhood = st.sidebar.selectbox('Neighbourhood',
                ('Brooklyn', 'Manhattan', 'Queens','Bronx','Staten Island'))
    state = st.sidebar.selectbox("State", ["New York"])
    country = st.sidebar.selectbox("Country", ["United States"])
    

    

    days_to_be_rented = st.sidebar.slider('Days in a Year Expected to Rent', 1,365, 1 )
    availability_365= st.sidebar.slider('Availability (must be greater than expected days to rent)', 0,365, 1)
    if availability_365 < days_to_be_rented:
        st.sidebar.warning("Please check that availability is greater than expected days to rent")
    else:
        st.sidebar.info('Ok!')
    room_type_option = st.sidebar.selectbox('Room Type',('Shared Room','Private Room','Entire house'))
    minimum_nights = st.sidebar.slider('Minimum Nights', 0,30, 1 )
    number_of_reviews = st.sidebar.slider('Number of reviews', 0,629, 1 )

    reviews_per_month = st.sidebar.slider('Reviews per month', 0,58, 1 )
    calculated_host_listings_count = st.sidebar.slider('Number of host listings', 1,327, 1 )
    

        
    sideb = st.sidebar


    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street+", "+neighborhood+", "+state+" ,"+country)


    try:
        latitude = location.latitude
        longitude = location.longitude
    except:
        st.error('There is an error with your location. Please check.')
        if street: # If user enters street, do 👇
            st.write(f'Please check that the {street} is actually in {neighborhood}!')
        #e = RuntimeError('This is an exception of type RuntimeError')
        #st.exception(e)
        #exit ()

    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

    st.map(map_data, zoom=15, use_container_width=True) 
  


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
            amenities_500 = 10
        with st.spinner('Calculating the number of leisure locations in a 500m radio...'):
            time.sleep(10)
        try:
            gdf_leisure = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_leisure) # Boundary to search within
        except:
            leisure_500 = 10
        with st.spinner('Calculating the number of train stations in a 500m radio...'):
            time.sleep(10)
        try: 
            gdf_subway= ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_subway)# Boundary to search within
        except:
            subway_500 = 10
        with st.spinner('Calculating the number of natural resources in a 500m radio...'):
            time.sleep(10)
        try:
            gdf_natural = ox.geometries.geometries_from_point((latitude, longitude),dist = 500, tags = tag_natural) # Boundary to search within
        except:
            natural_500 = 10
            
        def processs_all_geom(gdf_amenities, gdf_leisure,gdf_subway, gdf_natural):
            gdf_amenities['Center_point'] = gdf_amenities['geometry'].centroid
            gdf_leisure['Center_point'] = gdf_leisure['geometry'].centroid
            gdf_subway['Center_point'] = gdf_subway['geometry'].centroid
            gdf_natural['Center_point'] = gdf_natural['geometry'].centroid
            gdf_amenities["long"] = gdf_amenities.Center_point.map(lambda p: p.x)
            gdf_amenities["lat"] = gdf_amenities.Center_point.map(lambda p: p.y)
            gdf_leisure["long"] = gdf_leisure.Center_point.map(lambda p: p.x)
            gdf_leisure["lat"] = gdf_leisure.Center_point.map(lambda p: p.y)
            gdf_subway["long"] = gdf_subway.Center_point.map(lambda p: p.x)
            gdf_subway["lat"] = gdf_subway.Center_point.map(lambda p: p.y)
            gdf_natural["long"] = gdf_natural.Center_point.map(lambda p: p.x)
            gdf_natural["lat"] = gdf_natural.Center_point.map(lambda p: p.y)
            gdf_amenities['type'] = 'amenities'
            gdf_leisure['type'] = 'leisure'
            gdf_subway['type'] = 'subway'
            gdf_natural['type'] = 'natural'
            all_geom = pd.concat([gdf_amenities,gdf_leisure,gdf_subway,gdf_natural], ignore_index=True)
            all_geom_count = all_geom.groupby('type')['geometry'].count()
            all_geom_count = pd.DataFrame(all_geom_count).reset_index()
            return all_geom, all_geom_count
        
        all_geom, all_geom_count = processs_all_geom(gdf_amenities, gdf_leisure,gdf_subway, gdf_natural)
        fig = px.bar(all_geom_count, x='type', y='geometry')
        fig.show()
            
        st.success('Prediction Complete!')
        st.caption(f'Country Selected: {country}')
        st.caption(f'State Selected: {state}')
        st.caption(f'Neighborhood Selected: {neighborhood}')
        st.caption(f'Street Selected: {street}')
 
        
        
        
        amenities_500 = gdf_amenities.shape[0]
        st.write('Number of restaurants, airports, malls, hotels or pubs:')
        st.info(amenities_500)
        subway_500 = gdf_subway.shape[0]
        st.write(f'Number of Train Stations: {subway_500}')
        st.info(subway_500)
        natural_500 = gdf_natural.shape[0]
        st.write(f'Number of Baches and Parks: {natural_500}')
        st.info(natural_500)
        leisure_500 = gdf_leisure.shape[0]
        st.write(f'Number of  zoos, theme-parks, water-parks, and stadiums: {leisure_500}')
        st.info(leisure_500)
        user_data, tax_rate  = get_df(room_type_option,neighborhood, minimum_nights, number_of_reviews, reviews_per_month, calculated_host_listings_count, availability_365, amenities_500, leisure_500, subway_500, natural_500)
        
        pred, pred_upper, pred_lower = prediction(user_data)
        
        st.balloons()
        
        html_temp = """ 
        <div style ="background-color:gray;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Report</h1> 
        </div> 
        """
    
        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
        

        st.header('Price of Listing')
        st.info(f"Price Estimate per night: $ {str(np.round(pred[0], 2))}")
        st.info(f'The acceptable range is between:$ {str(np.round(pred_lower[0], 2))} and {str(np.round(pred_upper[0], 2))}')
        
        
        st.header('Revenue and Tax Rate Calculated')
        st.caption(f"Tax is calculated considering a tax rate in {neighborhood} of {tax_rate}. ")
        st.caption(f"Revenue is calculated assuming that the unit is rented for {days_to_be_rented} days in a year.")
        
        annual_revenue=np.round(pred[0]*120,2)
        calculate_tax=np.round(pred[0]*days_to_be_rented*tax_rate,2)
        col1, col2  = st.columns(2)
        col1.info("Revenue: $",str(annual_revenue))
        col2.info("Annual Assessed Tax: $", str(calculate_tax))

    
if __name__=='__main__': 
    main()


