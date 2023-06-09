# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2 - Singapore Housing Data and Kaggle Challenge

### Problem Statement

Housing Development Board (HDB) flats are a type of public housing in Singapore. They are highly sought after by both Singaporean citizens and permanent residents due to their affordability and various amenities provided by the government.

Real estate agents play a crucial role in facilitating the buying and selling of properties, including HDB flats. Their expertise and knowledge of the local market are essential in helping clients make informed decisions and achieve the best possible outcomes. To effectively serve their clients and maximize their business potential, real estate agents need accurate and reliable price predictions for HDB flats.

To address this need, our team of data scientists has been engaged by a group of real estate agents who are planning to establish their own real estate agency. They have specifically identified HDB flats as their primary focus and want us to develop a price prediction model tailored to this housing type. The aim of our data science project is to create a robust and accurate price prediction model for HDB flats. The successful implementation of this price prediction model will enable the real estate agents to make data-driven decisions, offer competitive pricing recommendations, attract more clients, and establish themselves as trusted advisors in the HDB market.

---

### Datasets

From Kaggle (https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price/data):
* [`test.csv`](../datasets/'test.csv'): Test data containing 77 predictor features for the model.
* [`train.csv`](../datasets/'train.csv'): Training data containing 77 predictor features and resale price for the model.

---

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**id**|*integer*|train_clean|unique id for each transaction|
|**tranc_yearmonth**|*string*|train_clean|year and month of the resale transaction, e.g. 2015-02|
|**town**|*string*|train_clean|HDB township where the flat is located, e.g. BUKIT MERAH|
|**flat_type**|*string*|train_clean|type of the resale flat unit, e.g. 3 ROOM|
|**block**|*string*|train_clean|block number of the resale flat, e.g. 454|
|**street_name**|*string*|train_clean|street name where the resale flat resides, e.g. TAMPINES ST 42|
|**storey_range**|*string*|train_clean|floor level (range) of the resale flat unit, e.g. 07 TO 09|
|**floor_area_sqm**|*float*|train_clean|floor area of the resale flat unit in square metres|
|**flat_model**|*string*|train_clean|HDB model of the resale flat, e.g. Multi Generation|
|**lease_commence_date**|*integer*|train_clean|commencement year of the flat unit's 99-year lease|
|**resale_price**|*float*|train_clean|the property's sale price in Singapore dollars|
|**tranc_year**|*integer*|train_clean|year of resale transaction|
|**tranc_month**|*integer*|train_clean|month of resale transaction|
|**mid_storey**|*integer*|train_clean|median value of storey_range|
|**lower**|*integer*|train_clean|lower value of storey_range|
|**upper**|*integer*|train_clean|upper value of storey_range|
|**mid**|*integer*|train_clean|middle value of storey_range|
|**full_flat_type**|*string*|train_clean|combination of flat_type and flat_model|
|**address**|*string*|train_clean|combination of block and street_name|
|**floor_area_sqft**|*float*|train_clean|floor area of the resale flat unit in square feet|
|**price_per_sqft**|*float*|train_clean|the property's price per square feet in Singapore dollars|
|**hdb_age**|*integer*|train_clean|number of years from lease_commence_date to present year|
|**max_floor_lvl**|*integer*|train_clean|highest floor of the resale flat|
|**year_completed**|*integer*|train_clean|year which construction was completed for resale flat|
|**residential**|*string*|train_clean|boolean value if resale flat has residential units in the same block|
|**commercial**|*string*|train_clean|boolean value if resale flat has commercial units in the same block|
|**market_hawker**|*string*|train_clean|boolean value if resale flat has a market or hawker centre in the same block|
|**multistorey_carpark**|*string*|train_clean|boolean value if resale flat has a multistorey carpark in the same block|
|**precinct_pavilion**|*string*|train_clean|boolean value if resale flat has a pavilion in the same block|
|**total_dwelling_units**|*integer*|train_clean|total number of residential dwelling units in the resale flat|
|**1room_sold**|*integer*|train_clean|number of 1-room residential units in the resale flat|
|**2room_sold**|*integer*|train_clean|number of 2-room residential units in the resale flat|
|**3room_sold**|*integer*|train_clean|number of 3-room residential units in the resale flat|
|**4room_sold**|*integer*|train_clean|number of 4-room residential units in the resale flat|
|**5room_sold**|*integer*|train_clean|number of 5-room residential units in the resale flat|
|**exec_sold**|*integer*|train_clean|number of executive type residential units in the resale flat block|
|**multigen_sold**|*integer*|train_clean|number of multi-generational type residential units in the resale flat block|
|**studio_apartment_sold**|*integer*|train_clean|number of studio apartment type residential units in the resale flat block|
|**1room_rental**|*integer*|train_clean|number of 1-room rental residential units in the resale flat block|
|**2room_rental**|*integer*|train_clean|number of 2-room rental residential units in the resale flat block|
|**3room_rental**|*integer*|train_clean|number of 3-room rental residential units in the resale flat block|
|**other_room_rental**|*integer*|train_clean|number of "other" type rental residential units in the resale flat block|
|**postal**|*string*|train_clean|postal code of the resale flat block|
|**latitude**|*float*|train_clean|Latitude based on postal code|
|**longitude**|*float*|train_clean|Longitude based on postal code|
|**planning_area**|*string*|train_clean|Government planning area that the flat is located|
|**mall_nearest_distance**|*float*|train_clean|distance (in metres) to the nearest mall|
|**mall_within_500m**|*float*|train_clean|number of malls within 500 metres|
|**mall_within_1km**|*float*|train_clean|number of malls within 1 kilometre|
|**mall_within_2km**|*float*|train_clean|number of malls within 2 kilometre|
|**hawker_nearest_distance**|*float*|train_clean|distance (in metres) to the nearest hawker centre|
|**hawker_within_500m**|*float*|train_clean|number of hawker centres within 500 metres|
|**hawker_within_1km**|*float*|train_clean|number of hawker centres within 1 kilometre|
|**hawker_within_2km**|*float*|train_clean|number of hawker centres within 2 kilometre|
|**hawker_food_stalls**|*integer*|train_clean|number of hawker food stalls in the nearest hawker centre|
|**hawker_market_stalls**|*integer*|train_clean|number of hawker and market stalls in the nearest hawker centre|
|**mrt_nearest_distance**|*float*|train_clean|distance (in metres) to the nearest MRT station|
|**mrt_name**|*string*|train_clean|name of the nearest MRT station|
|**bus_interchange**|*integer*|train_clean|boolean value if the nearest MRT station is also a bus interchange|
|**mrt_interchange**|*integer*|train_clean|boolean value if the nearest MRT station is a train interchange station|
|**mrt_latitude**|*float*|train_clean|latitude (in decimal degrees) of the the nearest MRT station|
|**mrt_longitude**|*float*|train_clean|longitude (in decimal degrees) of the nearest MRT station|
|**bus_stop_nearest_distance**|*float*|train_clean|distance (in metres) to the nearest bus stop|
|**bus_stop_name**|*string*|train_clean|name of the nearest bus stop|
|**bus_stop_latitude**|*float*|train_clean|latitude (in decimal degrees) of the the nearest bus stop|
|**bus_stop_longitude**|*float*|train_clean|longitude (in decimal degrees) of the nearest bus stop|
|**pri_sch_nearest_distance**|*float*|train_clean|distance (in metres) to the nearest primary school|
|**pri_sch_name**|*string*|train_clean|name of the nearest primary school|
|**vacancy**|*integer*|train_clean|number of vacancies in the nearest primary school|
|**pri_sch_affiliation**|*integer*|train_clean|boolean value if the nearest primary school has a secondary school affiliation|
|**pri_sch_latitude**|*float*|train_clean|latitude (in decimal degrees) of the the nearest primary school|
|**pri_sch_longitude**|*float*|train_clean|longitude (in decimal degrees) of the nearest primary school|
|**sec_sch_nearest_dist**|*float*|train_clean|distance (in metres) to the nearest secondary school|
|**sec_sch_name**|*string*|train_clean|name of the nearest secondary school|
|**cutoff_point**|*integer*|train_clean|PSLE cutoff point of the nearest secondary school|
|**affiliation**|*integer*|train_clean|boolean value if the nearest secondary school has an primary school affiliation|
|**sec_sch_latitude**|*float*|train_clean|latitude (in decimal degrees) of the the nearest secondary school|
|**sec_sch_longitude**|*float*|train_clean|longitude (in decimal degrees) of the nearest secondary school|

---

### Conclusion

1) Our production model can accurately and reliably predict HDB resale prices
2) Floor area, number of nearby hawkers, and floor level are the strongest positive predictors of resale prices
3) HDB age, location, and distance from MRT are the strongest negative predictors of resale prices
---

### Recommendations

1. To maximise commissions for the least effort, the agency may wish to leverage the prediction model and consider different strategies at the town, neighbourhood and block/unit levels. 

2. Emphasise on positive features during pitch to prospective sellers and buyers (i.e. maximising buyers/sellers’ returns) - consider a simple and user-friendly tool for prospective clients to play around with (try this at https://githubberst-project-2-streamlitapp2-856hmn.streamlit.app/)

3. Regularly provide content on features of interest (e.g. electronic direct mail, news and articles) to prospective clients

### Next steps

As our ageing population increases, how would this affect the demand for larger homes if homeowners retire, incomes are smaller and the children have all moved out? We can also further analyse and improve our prediction model with additional data such as age and gender of HDB owners. 





