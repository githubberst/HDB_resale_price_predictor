#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 07:55:43 2023

@author: wynne
"""

# imports
import streamlit as st
import pandas as pd
import numpy as np
import json
import folium as fs
import geopandas as gpd
import altair as alt
from streamlit_folium import st_folium, folium_static
import pickle

# first load the model and transformer

##this is for local drive##
# load the train model
# with open('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project-2/streamlit/or_model2.pkl', 'rb') as oridge:
#     model = pickle.load(oridge)

# # load the PowerTransformer
# with open('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project-2/streamlit/powertransformer.pkl', 'rb') as pt: 
#     transformer = pickle.load(pt) 
#     transformer.set_output(transform="pandas")

# # import data
# df = pd.read_csv('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project-2/datasets/train_pop_NaN.csv')

# # import the planning area boundaries
# with open ('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project-2/streamlit/master_plan_boundaries.json', 'r') as jsonFile:
#     town_borders = json.load(jsonFile)    

#this is for cloud deployment (toggle comment on for above when this is in use)##
# load the train model
with open('/app/project-2/streamlit/or_model2.pkl', 'rb') as oridge:
    model = pickle.load(oridge)

# load the PowerTransformer
with open('/app/project-2/streamlit/powertransformer.pkl', 'rb') as pt: 
    transformer = pickle.load(pt) 
    transformer.set_output(transform="pandas")
    
# import  data
df = pd.read_csv('/app/project-2/datasets/train_pop_NaN.csv')

# import the planning area boundaries
with open ('/app/project-2/streamlit/master_plan_boundaries.json', 'r') as jsonFile:
    town_borders = json.load(jsonFile)    
    

# some definitions and dictionaries

# list of flat types
types_list = [
    '1 ROOM Improved',
    '2 ROOM DBSS',
    '2 ROOM Improved',
    '2 ROOM Model A',
    '2 ROOM Premium Apartment',
    '2 ROOM Standard',
    '3 ROOM DBSS',
    '3 ROOM Improved',
    '3 ROOM Model A',
    '3 ROOM New Generation',
    '3 ROOM Premium Apartment',
    '3 ROOM Simplified',
    '3 ROOM Standard',
    '3 Room Terrace',
    '4 ROOM Adjoined flat',
    '4 ROOM DBSS',
    '4 ROOM Improved',
    '4 ROOM Model A',
    '4 ROOM Model A2',
    '4 ROOM New Generation',
    '4 ROOM Premium Apartment',
    '4 ROOM Premium Apartment Loft',
    '4 ROOM Simplified',
    '4 ROOM Standard',
    '4 Room Terrace',
    '4 Room Type S1',
    '5 ROOM Adjoined flat',
    '5 ROOM DBSS',
    '5 ROOM Improved',
    '5 ROOM Improved-Maisonette',
    '5 ROOM Model A',
    '5 ROOM Model A-Maisonette',
    '5 ROOM Premium Apartment',
    '5 ROOM Standard',
    '5 ROOM Type S2',
    '5 Room Premium Apartment Loft',
    'Executive Adjoined flat',
    'Executive Apartment',
    'Executive Maisonette',
    'Executive Premium Apartment',
    'Executive Premium Maisonette',
    'Multi-Generation']

# list of neighbourhoods

towns_list = [
    'Ang Mo Kio',
    'Bedok',
    'Bishan',
    'Bukit Batok',
    'Bukit Merah',
    'Bukit Panjang',
    'Bukit Timah',
    'Central area',
    'Choa Chu Kang',
    'Clementi',
    'Geylang',
    'Hougang',
    'Jurong East',
    'Jurong West',
    'Kallang/Whampoa',
    'Marine Parade',
    'Pasir Ris',
    'Punggol',
    'Queenstown',
    'Sembawang',
    'Seng Kang',
    'Serangoon',
    'Tampines',
    'Toa Payoh',
    'Woodlands',
    'Yishun']

# dictionaries with groupings that we determined based on our analysis and used in our 
# final model for prediction

towns_dict = {'grp1_town':['Bukit Timah','Marine Parade'], 
                  'grp2_town':['Queenstown','Bishan'], 
                  'grp3_town':['Clementi','Bukit Merah','Serangoon'], 
                  'grp4_town':['Punggol','Geylang','Tampines','Kallang/Whampoa','Central area','Toa Payoh'], 
                  'grp5_town':['Hougang','Bukit Batok','Bedok'], 
                  'grp6_town':['Jurong East','Pasir Ris'], 
                  'grp7_town':['Yishun','Seng Kang'], 
                  'grp8_town':['Choa Chu Kang','Jurong West','Bukit Panjang'], 
                  'grp9_town':['Sembawang','Woodlands']}

flat_type_dict = {'grp1_full_flat_type':['4 ROOM Terrace','3 ROOM Terrace'],
                    'grp2_full_flat_type':['4 ROOM Type S1','Multi-Generation','5 ROOM Premium Apartment Loft','5 ROOM DBSS','5 ROOM Improved-Maisonette','5 ROOM Type S2'], 
                    'grp3_full_flat_type':['4 ROOM New Generation','4 ROOM Simplified','5 ROOM Premium Apartment','4 ROOM Premium Apartment','5 ROOM Improved','4 ROOM Standard','Executive Premium Apartment'],
                    'grp4_full_flat_type':['3 ROOM DBSS','Executive Apartment','5 ROOM Standard'], 
                    'grp5_full_flat_type':['Executive Maisonette','5 ROOM Adjoined flat','4 ROOM Adjoined flat','4 ROOM DBSS'],
                    'grp6_full_flat_type':['Executive Premium Maisonette','4 ROOM Premium Apartment Loft','Executive Adjoined flat','5 ROOM Model A-Maisonette'],
                    'grp7_full_flat_type':['4 ROOM Model A','4 ROOM Improved','4 ROOM Model A2'],
                    'grp7_5_full_flat_type':['3 ROOM Simplified'], 
                    'grp8_full_flat_type':['3 ROOM Improved','3 ROOM Standard','3 ROOM Premium Apartment','3 ROOM Model A','3 ROOM New Generation'], 
                    'grp9_full_flat_type':['2 ROOM Standard','2 ROOM Model A','2 ROOM Improved','2 ROOM Premium Apartment'],
                    'grp10_full_flat_type':['1 ROOM Improved']}






# on to the good stuff


# main function 

def main():
    # streamlit shell (layouts etc)
    
    # set webpage name and icon
    st.set_page_config(
        page_title='Singapore HDB Price Predictor',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='collapsed'
        )
    
    
    # title, byline, and explanation
    st.title(':cityscape: HDB Price Predictor :cityscape:')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    st.text('Answer the following questions to get a prediction for the value of your flat.')
    st.text('Source: Data.gov.sg')
    style = "<div style='background-color:pink; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    # first section: map
    
    
    # group everything by town
    df3 = df.groupby('town').mean(numeric_only = True)
    
    
    # Latitude and longitude of every town in the list 
    lat = [1.369115,1.323604,1.352585,1.359029,1.281905,1.377414,1.329411,1.284484,1.38398,1.316181,1.320054,1.361218,1.332857,1.34039,1.324513,1.301969,1.372094,1.398446,1.294166,1.449111,1.386812,1.355357,1.349591,1.334304,1.438192,1.430368]
    lon = [103.845434,103.927341,103.835212,103.76368,103.823918,103.77195,103.802078,103.851345,103.746961,103.764938,103.891775,103.886253,103.743552,103.708988,103.857225,103.897082,103.947373,103.907205,103.786127,103.818495,103.891443,103.867871,103.956788,103.856327,103.78896,103.835363]
    
    #implementing the separate coordinates into the dataframe
    
    df3['latitude'] = lat
    df3['longitude'] = lon
    
    #implementing full coordinates into the dataframe
    coords = []
    for i,n in enumerate(df3.index):
        pt = [df3['latitude'][i],df3['longitude'][i]]
        coords.append(pt)
    
    df3['coordinates'] = coords
    df3['town'] = df3.index


    # creating a geodataframe
    geodf = gpd.GeoDataFrame(data = df3, geometry = gpd.points_from_xy(df3.longitude, df3.latitude), crs=4326)

    
    # defining the space for the map, with some buffer on the sides
    
    s, central, s2 = st.columns([1,300,1])
    
    with central:
        # now to bring in the map! it's centred on singapore's central catchment area.
        m = fs.Map(location=[1.363605, 103.814168],tiles = 'Stamen Toner', zoom_start=11)
        
        
        
        #with the overlay of the towns...
        
        choro_map =fs.Choropleth(geo_data = town_borders,
                                 data = geodf,
                                 columns = ['town','resale_price'],
                                 key_on = 'feature.properties.PLN_AREA_N',
                                 #pathway to locate each town and its geometry in the json file; it's a dict
                                 fill_color = 'YlGnBu', #yellow green blue
                                 nan_fill_color = "White", # color for empty data
                                 fill_opacity = 0.7, #density of colour
                                 line_opacity = 0.2,
                                 legend_name = 'Average Resale Price of HDB Flats',
                                 line_color = 'black',
                                 nan_fill_opacity = 0.3) #applies this choropleth to the map we called earlier
                                     
        
        geodf_index = geodf.set_index('town')
        
        
        # need to write the information to the geojson so we can map the info to the correct boundaries
        
        for i in choro_map.geojson.data['features']:
                if i['properties']['PLN_AREA_N'] not in list(df3['town']):  
                    i['properties']['resale_price'] = None
                else:
                    i['properties']['resale_price'] = geodf_index.loc[i['properties']['PLN_AREA_N'], 'resale_price']
        fs.GeoJsonTooltip(['PLN_AREA_N','resale_price']).add_to(choro_map.geojson)
        
        
       # call the markers we defined in a function below 
       
        markers(df3, m)
        
        choro_map.add_to(m)

        # print the map
        
        st_map = folium_static(m, width=1500)
        
        
    
    
    # second section: prediction inputs and button
    
    
        
    
    # primary dashboard for getting user inputs
    
    left, right = st.columns([2,2], gap = 'large')
    with left:
        st.subheader('About the flat:')
        hdb_age = st.number_input('How many years old is the flat (as of 2023)?',
                                        step=1, value=25)
        floor = st.number_input('Which floor is the flat on?',
                                    step=1, value=10) 
        floor_area_sqm = st.number_input('How many sqm is the flat?',
                                        step=1, value=100)
        input_flat_type = st.selectbox('What type of flat is it?', types_list, help='refer to explanation here')
        input_town = st.selectbox('Which area is the flat in?', towns_list)

        st.subheader('About the HDB block:')
        max_floor_lvl = st.number_input('How many floors are there in the block?',
                                    step=1, value=16)
        units_lvl = st.number_input('How many units are there on each floor?',
                                        step=1, value=8)
    with right:
        st.subheader('About nearby amenities:')
        st.write('Are the following amenities within 10min walk from the flat?')
        options = ['Yes','No']
        mrt = st.radio('MRT station', options, horizontal=True)
        bus="No" #setting bus_interchange as default No unless it meets below scenario
        if mrt == "Yes":
            bus = st.radio('Is this MRT station also a bus interchange?', options, horizontal=True)

        hawker = st.radio('Hawker centre', options, horizontal=True)
        mall = st.radio('Shopping mall', options, horizontal=True)
        psch = st.radio('Primary school', options, horizontal=True)
        ssch = st.radio('Secondary school', options, horizontal=True) 
        
        button = st.button('Predict resale price', help = 'Click to get your flat price prediction')
        
    
    
    #if button is pressed
    if button:
        
        # processing user input
        # this is where we do some tweaking to either fix values in place
        # or manipulate the user input for the model
        # the reason for this is to improve useability of the app
       
        
        # amenities time to distance conversion
        
        #to make the app more user friendly we simplified the question input to Y/N
        #However, what we actually need for our predictor is a distance. This function
        #converts 'time' to distance using the real estate rule of thumb that walking speed is
        #80m/min
         
        def amenities_distance(amenity):
            if amenity == "Yes": 
                return 10*80
            else: 
                return 20*80
        
        # flat storey level conversion to 'mid' 
        
        #we used 'mid' or the middle of the storey range as a feature, but to
        #simplify the app we ask the user what storey their flat is on.
        #this line of code re-assigns the closest floor in 'mid' to the user's floor
        #by iterating through mid and subtracting the user's input. 
        
        # hard coded mid values
        mid_floors = [2, 3, 5, 8, 11, 13, 14, 17, 18, 20, 23, 26, 28, 29, 32, 33, 35, 38, 41, 44, 47, 50]
        
        mid = min(mid_floors, key = lambda x: abs(x-floor)) 
        
        # hard coded to this year
        tranc_year = 2023
        
        # rather than ask how many flats there are in the block
        # we ask the user how many there are on their floor (should be easier)
        # and how high their flat is
        # and multiply it ourselves

        total_dwelling_units = units_lvl*max_floor_lvl
        
        # making use of the earlier amenities time-to-distance converter
        # on all the amenities

        mrt_nearest_distance = amenities_distance(mrt)
        hawker_nearest_distance = amenities_distance(hawker)
        mall_nearest_distance = amenities_distance(mall)
        pri_sch_nearest_distance = amenities_distance(psch)
        sec_sch_nearest_dist = amenities_distance(ssch)
        
        # converting Y/N to 1/0

        if bus == "Yes": 
            bus_interchange = 1
        else: 
            bus_interchange = 0 

        if hawker == "Yes": #assume within 2km as nearest hawker is within 10min
            hawker_within_2km = 1
        else: 
            hawker_within_2km = 0 
            
        # pre-set with median as people are unlikely to know 
        # how many market stalls are in the hawker near them
        
        hawker_market_stalls = 52 
        
        # group assignments for our modelling

        towngroup_assignments = {group: 1 if input_town in towns else 0 for group, towns in towns_dict.items()}
        grp1_town, grp2_town, grp3_town, grp4_town, grp5_town, grp6_town, grp7_town, grp8_town, grp9_town = towngroup_assignments.values()

        flattype_assignments = {group: 1 if input_flat_type in types else 0 for group, types in flat_type_dict.items()}
        grp1_full_flat_type, grp2_full_flat_type, grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type, grp6_full_flat_type, grp7_full_flat_type, grp7_5_full_flat_type, grp8_full_flat_type, grp9_full_flat_type, grp10_full_flat_type = flattype_assignments.values()
    
        
        
        #make prediction
        result = predict(floor_area_sqm, tranc_year, mid, hdb_age, max_floor_lvl, total_dwelling_units, mall_nearest_distance, 
                         hawker_nearest_distance, hawker_within_2km, hawker_market_stalls,
                         mrt_nearest_distance, bus_interchange, pri_sch_nearest_distance,
                         sec_sch_nearest_dist, grp1_town, grp2_town, grp3_town,
                         grp4_town, grp5_town, grp6_town, grp7_town, grp8_town,
                         grp9_town, grp1_full_flat_type, grp2_full_flat_type,
                         grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type,
                         grp6_full_flat_type, grp7_full_flat_type, grp8_full_flat_type,
                         grp9_full_flat_type, grp10_full_flat_type, grp7_5_full_flat_type)
        
        RMSE_final = 46000
        
        # escape needed because otherwise the double $s are read as LaTex values
        st.success(f'The value of this flat is between \${(result - RMSE_final)} to \${(result + RMSE_final)}.')
        
# define the prediction function 
# features in this function are the features we settled on in our final model

def predict(floor_area_sqm, tranc_year, mid, hdb_age, max_floor_lvl, total_dwelling_units, mall_nearest_distance, 
                 hawker_nearest_distance, hawker_within_2km, hawker_market_stalls,
                 mrt_nearest_distance, bus_interchange, pri_sch_nearest_distance,
                 sec_sch_nearest_dist, grp1_town, grp2_town, grp3_town,
                 grp4_town, grp5_town, grp6_town, grp7_town, grp8_town,
                 grp9_town, grp1_full_flat_type, grp2_full_flat_type,
                 grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type,
                 grp6_full_flat_type, grp7_full_flat_type, grp8_full_flat_type,
                 grp9_full_flat_type, grp10_full_flat_type, grp7_5_full_flat_type):
    
    input_list = [
        floor_area_sqm, tranc_year, mid, hdb_age, max_floor_lvl, \
        total_dwelling_units, mall_nearest_distance, hawker_nearest_distance, \
        hawker_within_2km, hawker_market_stalls, mrt_nearest_distance, \
        bus_interchange, pri_sch_nearest_distance, sec_sch_nearest_dist, \
        grp1_town, grp2_town, grp3_town, grp4_town, grp5_town, grp6_town, \
        grp7_town, grp8_town, grp9_town, grp1_full_flat_type, grp2_full_flat_type, \
        grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type, \
        grp6_full_flat_type, grp7_full_flat_type, grp8_full_flat_type, \
        grp9_full_flat_type, grp10_full_flat_type, grp7_5_full_flat_type
        ]
    df = pd.DataFrame(input_list).transpose()

    # logging those that need to be logged
    cols_to_log = [2, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    df.iloc[:, cols_to_log] = df.iloc[:, cols_to_log].applymap(lambda x: np.log2((x)+1))

    # scaling the data
    df=transformer.transform(df)

    # making predictions using the train model
    prediction = model.predict(df)
    
    # Reverse the y (resaleprice) since it is in the log form
    predicted = [round(2**val, 0) for val in prediction]
    #1.22 is obtained from calculation of %change between 1Q2023 and 1Q2021
    # (our end of train dataset) at https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics
    result = int(predicted[0]*1.22) 
    
    return result

def average_over_time(town):
    resale_dic = {}
    resale_town = df.loc[df['town'] == str(town)]
    
    i = 2012
    while i < 2022:
        resale_year = resale_town.loc[resale_town['tranc_year'] == i]
        resale_year_mean = round(resale_year['resale_price'].mean(), 2)
        resale_dic[str(i)] = resale_year_mean
        i += 1
    
      
     # filter off the NaN values inside the dic prior to sepearting them into lists 
    for key,value in resale_dic.items():
         if np.isnan(value) == True:
             del resale_dic[key]
             break
     
    # reindexing all the months + need the months for plotting
    # seperating the key value pairs into list for annotation
    resale_dic_key = list(resale_dic.keys())
    resale_dic_value = list(resale_dic.values())
    
    x = resale_dic_key
    
    # creating df to transfer them into altair
    source = pd.DataFrame({
        'Time': x,
        'Average Resale Price': resale_dic_value})
    
    # creating linegraph map using Altair.
    # hardcoding the axis scale for Y so that they all have the same scale
    a = alt.Chart(source,title= f'Average HDB Resale Price Of {town}').mark_line().encode( alt.X('Time:T'),
        y= alt.Y('Average Resale Price', scale=alt.Scale(domain=[300000, 900000]))
    ).properties(width = 300, height = 260) #remember 300 x300!
    
    return a 
 
# markers function implemented into the graph
def markers(df3, m):
    for i,n in enumerate(df3['town']):
        popup = fs.Popup(max_width = 400)
        a = average_over_time(n)
        v1 = a.to_json()
        marker_test  = fs.Marker(location = [df3['latitude'][i],df3['longitude'][i]],
                                 popup = popup.add_child(fs.VegaLite(v1,width = 900, height = 300)
                                                         ),tooltip = f'Click to see the average price over time!')
        marker_test.add_to(m)
    return m
     



# ensure main function runs immediately
if __name__ == '__main__':
    main()
    
   
