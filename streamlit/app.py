import pandas as pd
import numpy as np
import streamlit as st
import pickle

# load the train model
with open('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project02/streamlit/or_model2.pkl', 'rb') as oridge:
    model = pickle.load(oridge)

# load the PowerTransformer
with open('/Users/sandratan/Desktop/Sandra Copy (GithubGA)/GA projects/Project02/streamlit/powertransformer.pkl', 'rb') as pt:
    transformer = pickle.load(pt)
    transformer.set_output(transform="pandas")

# set webpage name and icon (shows up on tab)
st.set_page_config(
    page_title='Singapore HDB Prices',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed'
    )

def amenities_distance(amenity):
    if amenity == "Yes":
        return 10*80
    else:
        return 20*80

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
    '5 ROOM Model A'
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

def predict(floor_area_sqm, tranc_year, mid, hdb_age, max_floor_lvl,
       total_dwelling_units, mall_nearest_distance,
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
    result = "{:,}".format(int(predicted[0]*1.22)) #1.22 is obtained from calculation of %change between 1Q2023 and 1Q2021(our end of train dataset) at https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics

    return result

def main():
    buff1, col, buff2 = st.columns([1,3,1])
    
    with col:
        # title of webpage
        st.title('Predict Singapore HDB Resale Prices')
        style = "<div style='background-color:pink; padding:2px'></div>"
        st.markdown(style, unsafe_allow_html=True)
        left, right = st.columns([2,1])
        
        with left: 
            st.subheader('About your house:')
            hdb_age = st.number_input('How old is your house (in years, as of 2023)?',
                                            step=1, value=25)
            floor = st.number_input('Which floor is your house at?',
                                        step=1, value=10) 
            floor_area_sqm = st.number_input('How big is your house? (sqm)',
                                            step=1, value=100)
            input_flat_type = st.selectbox('What is the type of your house?', types_list, help='refer to explanation here')
            input_town = st.selectbox('Where is your house located at?', towns_list)

            st.subheader('About the HDB block:')
            max_floor_lvl = st.number_input('How many floors are there in the block?',
                                        step=1, value=16)
            units_lvl = st.number_input('How many units are there on your floor?',
                                            step=1, value=8)

            st.subheader('About nearby amenities:')
            st.write('Are the following amenities within 10min walk from your house?')
            options = ['Yes','No']
            mrt = st.radio('MRT station', options, horizontal=True)
            bus="No" #setting bus_interchange as default No unless it meets below scenario
            if mrt == "Yes":
                bus = st.radio('Is this MRT station also a bus interchange?', options, horizontal=True)
   
            hawker = st.radio('Hawker centre', options, horizontal=True)
            mall = st.radio('Shopping mall', options, horizontal=True)
            psch = st.radio('Primary school', options, horizontal=True)
            ssch = st.radio('Secondary school', options, horizontal=True)

        button = st.button('Predict resale price')
    
    # if button is pressed
    if button:

            # processing user input
        floors = [2, 3, 5, 8, 11, 13, 14, 17, 18, 20, 23, 26, 28, 29, 32, 33, 35, 38, 41, 44, 47, 50]
        prevfloor = -9999
        mid = -999
        for i in range(len(floors)):
            if floor == floors[i]:
                mid = floor #assign floor number as mid value since exact match
                break;
            if floor < floors[i]:
                if (floors[i] - floor) > (floor - prevfloor):
                    mid = prevfloor #assign the smaller mid value as the floor number is closer to it
                    break;
                else:
                    mid = floors[i] #else assign the bigger mid value 
                    break;
            prevfloor = floors[i]

        total_dwelling_units = units_lvl*max_floor_lvl

        mrt_nearest_distance = amenities_distance(mrt)
        hawker_nearest_distance = amenities_distance(hawker)
        mall_nearest_distance = amenities_distance(mall)
        pri_sch_nearest_distance = amenities_distance(psch)
        sec_sch_nearest_dist = amenities_distance(ssch)

        if bus == "Yes": 
            bus_interchange = 1
        else: 
            bus_interchange = 0 

        if hawker == "No": #assume as not within 2km when nearest hawker is not within 10min
            hawker_within_2km = 0
        else: 
            hawker_within_2km = 1 

        hawker_market_stalls = 52 #pre-set with median 

        towngroup_assignments = {group: 1 if input_town in towns else 0 for group, towns in towns_dict.items()}
        grp1_town, grp2_town, grp3_town, grp4_town, grp5_town, grp6_town, grp7_town, grp8_town, grp9_town = towngroup_assignments.values()

        flattype_assignments = {group: 1 if input_flat_type in types else 0 for group, types in flat_type_dict.items()}
        grp1_full_flat_type, grp2_full_flat_type, grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type, grp6_full_flat_type, grp7_full_flat_type, grp7_5_full_flat_type, grp8_full_flat_type, grp9_full_flat_type, grp10_full_flat_type = flattype_assignments.values()
    
        # predict 
        tranc_year = 2023 #pre-set as the year of transaction 
        result = predict(floor_area_sqm, tranc_year, mid, hdb_age, max_floor_lvl, total_dwelling_units, mall_nearest_distance, 
                         hawker_nearest_distance, hawker_within_2km, hawker_market_stalls,
                         mrt_nearest_distance, bus_interchange, pri_sch_nearest_distance,
                         sec_sch_nearest_dist, grp1_town, grp2_town, grp3_town,
                         grp4_town, grp5_town, grp6_town, grp7_town, grp8_town,
                         grp9_town, grp1_full_flat_type, grp2_full_flat_type,
                         grp3_full_flat_type, grp4_full_flat_type, grp5_full_flat_type,
                         grp6_full_flat_type, grp7_full_flat_type, grp8_full_flat_type,
                         grp9_full_flat_type, grp10_full_flat_type, grp7_5_full_flat_type)
        
        st.success(f'The value of the house is ${result}')

if __name__ == '__main__':
    main()