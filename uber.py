from uber_rides.session import Session
from uber_rides.client import UberRidesClient

# creating session
session = Session(server_token=<TOKEN>)
client = UberRidesClient(session)

# requesting estimation of price
response = client.get_price_estimates(
    start_latitude=37.770,
    start_longitude=-122.411,
    end_latitude=37.791,
    end_longitude=-122.405,
    seat_count=2
)

estimate = response.json.get('prices')

# Get products for a location
response = client.get_products(37.77, -122.41)
products = response.json.get('products')

product_id = products[0].get('product_id')

# Get upfront fare and start/end locations
estimate = client.estimate_ride(
    product_id=product_id,
    start_latitude=37.77,
    start_longitude=-122.41,
    end_latitude=37.79,
    end_longitude=-122.41,
    seat_count=2
)
fare = estimate.json.get('fare')

# Request a ride with upfront fare and start/end locations
response = client.request_ride(
    product_id=product_id,
    start_latitude=37.77,
    start_longitude=-122.41,
    end_latitude=37.79,
    end_longitude=-122.41,
    seat_count=2,
    fare_id=fare['fare_id']
)

request = response.json
request_id = request.get('request_id')

# Request ride details using `request_id`
response = client.get_ride_details(request_id)
ride = response.json

# Cancel a ride
response = client.cancel_ride(request_id)
ride = response.json