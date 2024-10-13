import python_weather
import csv
import asyncio
import os

async def getweather() -> None:
    # Open the CSV file and add headers
    with open('Data/hourly_weather_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Time', 'Temperature', 'Condition'])

        # Declare the client and fetch weather data
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            weather = await client.get('Boston')
            
            # Print the complete structure of the weather object for inspection
            print("Weather object:", weather)

            # Write current weather data to CSV
            if hasattr(weather, 'temperature'):
                date_str = weather.datetime.strftime('%Y-%m-%d')
                time_str = weather.datetime.strftime('%H:%M')
                temperature = weather.temperature
                condition = "N/A"  # Placeholder for current condition
                
                # Write current data to CSV
                writer.writerow([date_str, time_str, temperature, condition])
                # print(f'Written current data to CSV: {date_str} {time_str} -> {temperature}°F, {condition}')

            # Check and write daily forecast data if available
            for daily in weather:
                for hourly in daily:
                    # print(hourly)
                    writer.writerow([daily.date, hourly.time, hourly.temperature, hourly.description, hourly.kind])
                    # print(f' --> {hourly!r}')

            else:
                print("No daily forecast data available.")

    print("Data written to 'hourly_weather_data.csv'. Please check the file.")

if __name__ == '__main__':
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(getweather())
