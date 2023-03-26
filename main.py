# %%

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import Geohash
import numpy as np

# %%

#Append all the json files to one dataframe
data_directory = os.getcwd() + '\data'

df = pd.DataFrame

for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    # checking if it is a file
    if os.path.isfile(file_path):
        #print(file_path)
        with open(file_path, 'r') as file:
            data = json.loads(file.read())
        #flatten data
        flatten_data = pd.json_normalize(data, record_path = ['data'])
        #Add to df
        if df.empty:
            df = flatten_data
        else:
            df = pd.concat([df, flatten_data], ignore_index=True)

        break
        
df.to_csv('dataframe_output/dataframe.csv')

# %% Port Area Definition
min_latitude = 51.8
max_latitude = 52.1
min_longitude = 3.8
max_longitude = 4.7

# Corner points of the port area
area_lats = [min_latitude, min_latitude, max_latitude, max_latitude, min_latitude]
area_lons = [min_longitude, max_longitude, max_longitude, min_longitude, min_longitude]

# Select only AIS data from within Rotterdam port area
df2 = df.loc[(df['navigation.location.lat']>= min_latitude) & (df['navigation.location.lat']<= max_latitude) & (df['navigation.location.long']>= min_longitude) & (df['navigation.location.long']<= max_longitude)] 


# %% Convex hull
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# create list with all coordinates 
coordinates = df2[['navigation.location.lat', 'navigation.location.long']].to_numpy()

hull = ConvexHull(coordinates) 

# get the coordinates of the corner points
corner_points = coordinates[hull.vertices]

# make a list of the hull coordinates
hull_lats = []
hull_lons = []

for i in corner_points:
    hull_lats.append(i[0])
    hull_lons.append(i[1])

# add strating coordinates to enclose the hull area
hull_lats.append(corner_points[0][0])
hull_lons.append(corner_points[0][1])


# %% Convex hull area

def reproject(latitude, longitude):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0

    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat)) 
                for lat, long in zip(latitude, longitude)]
    return x, y

hull_lats_reprojected, hull_lons_reprojected = reproject(hull_lats, hull_lons)

def area_of_polygon(x, y):
    """Calculates the area of an arbitrary polygon given its verticies"""
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0

convex_hull_area = area_of_polygon(hull_lats_reprojected, hull_lons_reprojected)


# %% Geohash 
#add geohash column to data frame, precision 7
for index, row in df2.iterrows():
    df2.loc[index, 'geohash'] = Geohash.encode(df2.loc[index, 'navigation.location.lat'], df2.loc[index, 'navigation.location.long'])[0:7]


# %% Gehash area

geohash_list = []

for item in df2['geohash']:
    if item not in geohash_list:
        geohash_list.append(item)

# Geohash area with precision 7 is equal to 153m * 153m

geohash_area = len(geohash_list) * (153*153)

print('geohash_area covers ', geohash_area / convex_hull_area, ' percent of hull area' )


# %% Plot (filtered) data
# plotly plot opens in browser
if True:
    # AIS data
    color_scale = [(0, 'orange'), (1,'red')]
    fig = px.scatter_mapbox(df2, 
                            lat="navigation.location.lat", 
                            lon="navigation.location.long", 
                            hover_name="vessel.name", 
                            hover_data=["vessel.name", "navigation.time"],
                            color="vessel.name",
                            color_continuous_scale=color_scale,
                            zoom=8, 
                            height=800,
                            width=1600)
    
    # Port area
    fig2 = px.line_mapbox(lat=area_lats, lon=area_lons, color_discrete_sequence=["black"])

    # Hull area
    fig3 = px.line_mapbox(lat=hull_lats, lon=hull_lons)

    # Add traces to fig 1
    fig.add_trace(fig2.data[0]) # adds the line trace to the first figure
    fig.add_trace(fig3.data[0]) # adds the line trace to the first figure

    # Update layout
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig.show()

# %%
