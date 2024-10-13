import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Function to generate prices with peaks during rush hours (5 PM to 7 PM)
def generate_prices_rush_hours():
    # Define the time range from 10 AM to 10 PM (12 hours)
    start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    times = [start_time + timedelta(minutes=i * 10) for i in range(72)]  # 10-minute intervals
    
    base_price = 20  # Base price around which we vary the prices
    
    # Define rush hour time ranges (5 PM to 7 PM)
    rush_hour_start = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    rush_hour_end = datetime.now().replace(hour=19, minute=0, second=0, microsecond=0)
    
    # Generate prices with slight differences between Uber, Lyft, and Wingz
    def generate_prices_with_rush(base_price, variation_range):
        prices = []
        for time in times:
            # Higher price volatility during rush hours
            if rush_hour_start <= time <= rush_hour_end:
                variation = np.random.uniform(variation_range[0], variation_range[1])  # Bigger variation during rush hours
            else:
                variation = np.random.uniform(-0.5, 0.5)  # Very small variation during non-rush hours
            prices.append(base_price + variation)
        
        return prices

    # Generate prices for Wingz (base price), then adjust Lyft and Uber
    wingz_prices = generate_prices_with_rush(base_price, (1, 2))  # Base price for Wingz
    
    # Lyft prices: $1-$3 more than Wingz, but fluctuates
    lyft_prices = [price + np.random.uniform(1, 3) for price in wingz_prices]
    
    # Uber prices: $1-$3 more than Lyft, but fluctuates
    uber_prices = [price + np.random.uniform(1, 3) for price in lyft_prices]
    
    return times, uber_prices, lyft_prices, wingz_prices

# Streamlit app interface
st.title("Ride-Hailing Price Comparison (10 AM to 10 PM)")

# Generate price data from 10 AM to 10 PM with rush hour spikes
times, uber_prices, lyft_prices, wingz_prices = generate_prices_rush_hours()

# Plotting the price vs time graph
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(times, uber_prices, label="Uber", color="black", linestyle='-', linewidth=2)
ax.plot(times, lyft_prices, label="Lyft", color="purple", linestyle='-', linewidth=2)
ax.plot(times, wingz_prices, label="Wingz", color="blue", linestyle='-', linewidth=2)



# Customize the plot
ax.set_xlabel('Time')
ax.set_ylabel('Price ($)')
ax.set_title('Price Comparison of Ride-Hailing Services (10 AM to 10 PM with Rush Hours)')
ax.legend()

# Format the x-axis to show the time for the full day (10 AM to 10 PM)
ax.set_xticks([times[i] for i in range(0, len(times), 6)])  # Show time every hour
ax.set_xticklabels([t.strftime('%I:%M %p') for t in times[::6]], rotation=45)  # Format as AM/PM

# Show the plot in Streamlit
st.pyplot(fig)
