import pandas as pd
import numpy as np

# Load the CSV file into a pandas DataFrame (update the file path)
uber_df = pd.read_csv("uber.csv")

# Function to calculate the Haversine distance between two points on the Earth
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    return c * r  # Distance in kilometers

# Step 1: Calculate the distance between pickup and dropoff points
uber_df['distance_km'] = haversine(
    uber_df['pickup_latitude'], 
    uber_df['pickup_longitude'], 
    uber_df['dropoff_latitude'], 
    uber_df['dropoff_longitude']
)

# Step 2: Drop the unnamed first column and coordinate columns
uber_df_cleaned = uber_df.drop(columns=['Unnamed: 0', 'key', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count'])

# Step 3: Modify the 'pickup_datetime' column to only display the time (remove date)
uber_df_cleaned['pickup_time'] = pd.to_datetime(uber_df_cleaned['pickup_datetime']).dt.time

# Drop the original 'pickup_datetime' column
uber_df_cleaned = uber_df_cleaned.drop(columns=['pickup_datetime'])

#Step 3.5: Remove any rows with missing values:
uber_df_cleaned = uber_df_cleaned.dropna()

# Step 4: Calculate the average distance
average_distance = uber_df_cleaned['distance_km'].mean()

# Step 5: Calculate the average price (if 'price' column exists)
if 'fare_amount' in uber_df_cleaned.columns:
    average_price = uber_df_cleaned['fare_amount'].mean()
else:
    average_price = "Price column not found in the dataset."

# Print the averages
print(f"Average Distance (km): {average_distance}")
print(f"Average Price: {average_price}")
print(uber_df_cleaned)  

# Optionally, save the cleaned DataFrame to a new CSV file
# uber_df_cleaned.to_csv('uber_cleaned_with_distance_and_time.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Assuming that uber_df_cleaned is already available from your preprocessing steps
# Preprocessing: Extract hour from pickup_time and add it as a new feature
uber_df_cleaned['pickup_hour'] = pd.to_datetime(uber_df_cleaned['pickup_time'], format='%H:%M:%S').dt.hour

# Step 1: Prepare features (distance_km and pickup_hour) and target (fare_amount)
X = uber_df_cleaned[['distance_km', 'pickup_hour']]
y = uber_df_cleaned['fare_amount']

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train)

# Step 4: Make predictions
y_pred_lin = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)
y_pred_gb = gb_reg.predict(X_test)

# Step 5: Evaluate models

def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print("-" * 40)

# Evaluate Linear Regression
evaluate_model(y_test, y_pred_lin, "Linear Regression")

# Evaluate Random Forest
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Evaluate Gradient Boosting
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Example: Predict the fare for a trip
# Assume the trip has a distance of 10 km and happens at 3 PM (15:00)
new_trip = pd.DataFrame({
    'distance_km': [5],    # Example distance in kilometers
    'pickup_hour': [11]     # 3 PM in 24-hour format
})

# Use the trained Gradient Boosting model to predict the fare
predicted_fare = gb_reg.predict(new_trip)

# Output the predicted fare
print(f"Predicted fare for a 5 Mile trip at 11 AM: {predicted_fare[0]:.2f}")