import urllib3

# Create a PoolManager instance to handle HTTP requests
http = urllib3.PoolManager()

# Define the headers
headers = {
    'Authorization': 'TOKEN EIUexiuatseoiu3u51351iiuiou35252'
}

# Make the request
url = 'http://polls.apiblueprint.org/v3/estimates/fare?pickup_latitude=38.840698&pickup_longitude=77.063103&dropoff_latitude=38.920266&dropoff_longitude=77.041580&pickup_time=1578942841'
response = http.request('GET', url, headers=headers)

# Decode the response body to a string
response_body = response.data.decode('utf-8')
print(response_body)
