import requests
import csv

# Specify the API endpoint and parameters
url = 'https://services.marinetraffic.com/api/exportvesseltrack/cbb81e50f6a05bfaa4ddaaec68f4ccf4e1c4e309'
params = {
    'timespan': '24',
    'start_date': '2022-01-01 00:00:00',
    'end_date': '2022-01-02 00:00:00',
    'msgtype': '4',
    'protocol': 'json',
    'mmsi': '',
    'imo': '',
    'callsign': '',
    'vessel_name': '',
    'port_id': '1386',
    'area_min_lat': '51.8279',
    'area_max_lat': '51.9925',
    'area_min_lon': '3.7257',
    'area_max_lon': '4.4928'
}

# Make a request to the API and retrieve the data
response = requests.get(url, params=params)
data = response.json()
print(data[0])  # print the first row of data to inspect its structure

# Save the data to a CSV file
with open('ais_data_rotterdam.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'latitude', 'longitude', 'speed', 'course', 'mmsi', 'imo', 'name', 'callsign', 'type'])
    for row in data:
        writer.writerow([row['TIMESTAMP'], row['LAT'], row['LON'], row['SPEED'], row['COURSE'], row['MMSI'], row['IMO'], row['NAME'], row['CALLSIGN'], row['TYPE']])
