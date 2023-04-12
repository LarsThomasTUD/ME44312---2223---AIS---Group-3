# %%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.io as pio
import Geohash
import numpy as np
from datetime import datetime
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import math
import random

# %% Append all the json files to one dataframe

if os.path.exists('dataframe_output/dataframe.csv'):
    df = pd.read_csv('dataframe_output/dataframe.csv')
else:
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
    df.to_csv('dataframe_output/dataframe.csv')

# %% Data filters
# Port Area Definition
min_latitude = 51.8
max_latitude = 52.1
min_longitude = 3.8
max_longitude = 4.7

# Corner points of the port area
area_lats = [min_latitude, min_latitude, max_latitude, max_latitude, min_latitude]
area_lons = [min_longitude, max_longitude, max_longitude, min_longitude, min_longitude]

# Select only AIS data from within Rotterdam port area
df_filtered = df.loc[(df['navigation.location.lat']>= min_latitude) & (df['navigation.location.lat']<= max_latitude) & (df['navigation.location.long']>= min_longitude) & (df['navigation.location.long']<= max_longitude)] 

# Add geohash
# add geohash column to data frame, precision 7
for index, row in df_filtered.iterrows():
    df_filtered.loc[index, 'geohash'] = Geohash.encode(df_filtered.loc[index, 'navigation.location.lat'], df_filtered.loc[index, 'navigation.location.long'])[0:7]

# %% Create snapshot
snapshot_names = []
for i in range(1, 3):
    snapshot_names.append('base_snapshot_' + str(i))

#snapshot_names.append('snapshot_not_congested')
#snapshot_names.append('snapshot_congested')

snapshot_dfs = {}
for snapshot_df in snapshot_names:
    snapshot_name = snapshot_df
    snapshot_location = 'snapshot_data/base/' + snapshot_name + '.csv'
    #Only create a new snapshot if it does not exist already 
    if os.path.exists(snapshot_location):
        snapshot_df = pd.read_csv(snapshot_location)
        # Reset the index of the new dataframe
        snapshot_df = snapshot_df.reset_index(drop=True)
    else:
        # Determine the number of rows to select randomly
        num_rows = random.randint(100, 600)
        snapshot_df = df_filtered.sample(n=num_rows)
        # Reset the index of the new dataframe
        snapshot_df = snapshot_df.reset_index(drop=True)
        # Write snapshot to snapshot_data file 
        snapshot_df.to_csv(snapshot_location)

    snapshot_dfs[snapshot_name] = snapshot_df


for i in (100, 200, 300, 400, 500, 600):
    snapshot_names_experiment = []
    for j in range(1, 3):
        snapshot_names_experiment.append('experiment_' + str(i) + '_snapshot_' + str(j))
    for snapshot_df in snapshot_names_experiment:
        snapshot_name = snapshot_df
        snapshot_location = 'snapshot_data/experiment_' + str(i) + '/' + snapshot_name + '.csv'
        #Only create a new snapshot if it does not exist already 
        if os.path.exists(snapshot_location):
            snapshot_df = pd.read_csv(snapshot_location)
            # Reset the index of the new dataframe
            snapshot_df = snapshot_df.reset_index(drop=True)
        else:
            # Determine the number of rows to select randomly
            num_rows = i
            snapshot_df = df_filtered.sample(n=num_rows)
            # Reset the index of the new dataframe
            snapshot_df = snapshot_df.reset_index(drop=True)
            # Write snapshot to snapshot_data file 
            snapshot_df.to_csv(snapshot_location)

        snapshot_dfs[snapshot_name] = snapshot_df



# %% Iterate for each snapshot
runs = ['base', 'experiment_100', 'experiment_200']
for run in runs:
    print(run)
    # specify the path of the location you want to search
    path = '/snapshot_data/' + run + '/'

    # use a list comprehension to get all CSV filenames in the directory
    csv_files = [f for f in os.listdir(os.getcwd() + path) if f.endswith('.csv')]

    snapshot_dfs = {}
    snapshot_names = []

    # print the CSV filenames
    for csv_file in csv_files:
        snapshot_names.append(csv_file.replace('.csv', ''))
        snapshot_df = pd.read_csv('snapshot_data/' + run + '/' + csv_file)
        # Reset the index of the new dataframe
        snapshot_df = snapshot_df.reset_index(drop=True)
        snapshot_dfs[csv_file.replace('.csv', '')] = snapshot_df


    # First create a return dataframe with the results
    results = pd.DataFrame({'Snapshot': snapshot_names})
    results.set_index('Snapshot', inplace=True)

    for snapshot_name, snapshot_df in snapshot_dfs.items():
        # create list with all coordinates 
        coordinates = snapshot_df[['navigation.location.lat', 'navigation.location.long']].to_numpy()
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

        # Convex hull area
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
        results.loc[snapshot_name, 'convex_hull_area'] = convex_hull_area

        # Average vessel prosimity 
        proximity_coordinates = snapshot_df[['navigation.location.lat', 'navigation.location.long']].to_numpy()

        def haversine(lat1, lon1, lat2, lon2):
            """
            Calculate the distance between two points on the surface of a sphere using the Haversine formula.
            """
            R = 6371009  # Earth's radius in meters

            # Convert latitude and longitude coordinates from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            # Calculate the differences between the latitudes and longitudes
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Calculate the Haversine formula
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            return distance

        # Calculate the distances between all pairs of coordinates
        distances = []
        for i in range(len(proximity_coordinates)):
            for j in range(len(proximity_coordinates)):
                if i != j:
                    distance = haversine(proximity_coordinates[i][0], proximity_coordinates[i][1], proximity_coordinates[j][0], proximity_coordinates[j][1])
                    distances.append(distance)

        # Calculate the average proximity
        average_proximity = sum(distances) / len(distances) 
        results.loc[snapshot_name, 'average_proximity'] = average_proximity

        # Geohash area
        geohash_list = []
        for item in snapshot_df['geohash']:
            if item not in geohash_list:
                geohash_list.append(item)

        # Geohash area with precision 7 is equal to 153m * 153m
        geohash_area = len(geohash_list) * (153*153)
        results.loc[snapshot_name, 'geohash_area'] = geohash_area

        # Plot (filtered) data
        # plotly plot opens in browser
        if True:
            # Remove vessel.name as index
            snapshot_df = snapshot_df.reset_index()

            # AIS data
            color_scale = [(0, 'orange'), (1,'red')]
            snapshot_df['dummy_column_for_size'] = 1.

            fig = ff.create_hexbin_mapbox(
                data_frame=snapshot_df, 
                lat="navigation.location.lat", 
                lon="navigation.location.long",  
                nx_hexagon=20, 
                opacity=0.8, 
                zoom=8, 
                height=800,
                width=1600, 
                labels={"color": "Vessel Count"},
                min_count=1  # set min_count to 1 to hide empty hexagons
            )
            
            # Port area
            fig2 = px.line_mapbox(lat=area_lats, lon=area_lons, color_discrete_sequence=["black"])

            # Hull area
            fig3 = px.line_mapbox(lat=hull_lats, lon=hull_lons)

            # Add traces to fig 1
            fig.add_trace(fig2.data[0]) # adds the line trace to the first figure
            fig.add_trace(fig3.data[0]) # adds the line trace to the first figure

            # Update layout
            fig.update_layout(
                    title=str(snapshot_name + '; ' + str(len(snapshot_df)) + ' vessel locations, ' + str(convex_hull_area) + 'm2 convex hull area, ' + str(geohash_area) + 'm2 geohash area, ' + str(average_proximity) + 'm average vessel proximity ') ,
                    mapbox=dict(
                        accesstoken='your_token_here',
                        style='open-street-map', # set mapbox style here
                        center=dict(lat=51.951897, lon=4.263340),
                        zoom=10 # set zoom level here
                    ),
                    margin={"r":20,"t":40,"l":20,"b":0} # set margin here
                )

            # Save figure
            fig.write_image(os.getcwd() + '/output/' + run + '/figures/' + str(snapshot_name) + '.png', scale=4)
            #fig.show()

    # 
    results['SpComplexity'] = 0
    results['SpDensity'] = 0

    for i in range(len(results)):
        results['SpComplexity'][i] = (1 / results['average_proximity'][i]) * (1 / results['convex_hull_area'][i])
        results['SpDensity'][i] = (1 / results['geohash_area'][i]) * (1 / results['convex_hull_area'][i])

    max_SpComplexity = results['SpComplexity'].max()
    max_SpDensity = results['SpDensity'].max()

    for i in range(len(results)):
        results['SpComplexity'][i] = results['SpComplexity'][i] / max_SpComplexity
        results['SpDensity'][i] = results['SpDensity'][i] / max_SpDensity

    # Save results to output location 
    results.to_csv(os.getcwd() + '/output/' + run + '/results/' + run + '_results.csv', index=True)


# %%
    # Clustering
    SpComplexity = results['SpComplexity']
    SpDensity = results['SpDensity']


    # Create an array of shape (n_samples, 2) where the first column is SpComplexity and the second column is SpDensity
    X = np.column_stack((SpComplexity, SpDensity))

    # Create a KMeans object with K=3 clusters
    kmeans = KMeans(n_clusters=3)

    # Fit the KMeans model to the data
    kmeans.fit(X)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Get the centroids of each cluster
    centroids = kmeans.cluster_centers_ 

    # Create a DataFrame with the data points and their labels
    results['Cluster'] = labels

    # Create a scatter plot using Plotly Express
    fig = px.scatter(results, x='SpComplexity', y='SpDensity', color='Cluster', 
                    title='K-means clustering', hover_name = results.index, hover_data=['Cluster'], 
                    template='plotly_white')

    # Add black crosses for the centroids
    fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', 
                    marker=dict(symbol='x', size=10, color='black'))

    # Show the plot
    fig.show()
    # %%
