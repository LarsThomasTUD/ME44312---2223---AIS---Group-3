import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


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

        #break
        


# Dataframe information
print(df)

df.info()


# plot data
# plotly plot opens in browser
color_scale = [(0, 'orange'), (1,'red')]

fig = px.scatter_mapbox(df, 
                        lat="navigation.location.lat", 
                        lon="navigation.location.long", 
                        hover_name="vessel.name", 
                        hover_data=["vessel.name", "navigation.time"],
                        color="vessel.name",
                        color_continuous_scale=color_scale,
                        zoom=8, 
                        height=800,
                        width=1600)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

