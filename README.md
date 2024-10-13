# HackHarvard2024: Transparent Ride-Hailing Price Discrepancy Solution

## Project Overview
We developed a revolutionary, cost-saving, and transparent solution to tackle price discrepancies and volatile ride-hailing fares by leveraging Uber data. Our solution integrates seamlessly with Google Maps, allowing consumers to view real-time price estimates and identify the best fare options directly when booking a ride. This project empowers consumers to make informed decisions and save money, especially during peak hours and surge pricing periods.

## Tech Stack
Our solution leverages a modern tech stack that includes:

- **Python**: Core language for data manipulation and backend logic.
- **TensorFlow**: Utilized to build and train machine learning models for predicting future fare prices.
- **FastAPI**: Framework used for building the backend API, enabling fast, real-time price estimates.
- **Google Maps API**: Integrated to provide a seamless user experience, allowing users to visualize fare prices directly on a map.
- **Common Libraries**: We used libraries such as `pandas` and `numpy` for data preprocessing and analysis, along with `scikit-learn` for model evaluation and train/test splits.

## Machine Learning Approach

To predict ride-hailing prices and optimize savings, we experimented with various machine learning algorithms and models, including:

- **Linear Regression**: For baseline predictions.
- **Random Forest Regressor**: To handle non-linear relationships between factors like time, distance, and demand.
- **Neural Networks**: Built using TensorFlow to capture more complex patterns in the data for accurate predictions, especially during volatile periods like rush hours.

We performed a thorough train/test split (80/20) to evaluate the effectiveness of each model and fine-tuned hyperparameters for optimal performance.

### Model Training
1. **Data Preprocessing**: Cleaned and preprocessed the dataset from Uber, removing outliers and handling missing values. Feature engineering included adding time-of-day and day-of-week factors to account for surge pricing periods.
   
2. **Training and Validation**: We tested multiple models, starting with a simple Linear Regression model, moving to more sophisticated Random Forest models, and finally implementing a deep learning approach using TensorFlow.

3. **Evaluation**: The final model was selected based on a combination of Mean Squared Error (MSE) and cross-validation scores, ensuring accuracy and generalizability.

## How It Works
- **Data Collection**: Using Uber’s historical ride data, the model predicts fare prices based on factors such as time of day, location, and demand spikes.
- **Real-Time Price Estimates**: Consumers can easily view ride fare predictions directly through Google Maps by selecting a location. The application displays predicted prices for Uber, Lyft, and other services, giving users the ability to compare prices before booking.
- **Cost Savings**: By predicting price volatility and discrepancies, the tool helps consumers save money, especially during peak travel hours.

## Challenges and Solutions
During development, we faced several challenges:
1. **Data Sparsity During Off-Peak Hours**: The data from Uber was sparse during off-peak hours, leading to potential overfitting. To address this, we used data augmentation techniques and applied a Random Forest Regressor to smooth the price predictions.
   
2. **API Rate Limits**: Integrating Google Maps API led to rate-limiting issues when pulling large volumes of location data. We implemented request caching and optimized API calls to stay within limits.

3. **Model Overfitting**: Initially, our TensorFlow model overfitted to the training data due to the complexity of the neural network. We reduced this by introducing dropout layers and regularization techniques during training.

## Future Improvements
- **User Personalization**: Adding user-specific fare recommendations based on their previous ride patterns.
- **Expanded Service Providers**: Including more ride-hailing services, such as local cab companies and car-sharing platforms.
- **Dynamic Price Updates**: Real-time updates from multiple service providers based on immediate market conditions.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/oolongmilktea-a/HackHarvard2024.git
