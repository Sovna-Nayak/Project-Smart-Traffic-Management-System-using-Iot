import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Traffic Prediction and Congestion Hotspot Detection Script
- Generates synthetic traffic data simulating hourly traffic volumes for multiple locations
- Trains a RandomForest regressor to predict future traffic volume trends
- Detects congestion hotspots based on predicted traffic volumes exceeding a threshold
- Visualizes traffic trends and hotspots

This script demonstrates a scalable, cost-effective ML approach for traffic optimization
"""

def generate_synthetic_traffic_data(num_locations=5, days=30):
    """
    Generate synthetic hourly traffic volume data for multiple locations over a number of days.
    Traffic volumes have daily and weekly patterns plus some randomness.
    """
    np.random.seed(42)
    hours = days * 24
    data = []
    for loc in range(num_locations):
        base_volume = np.random.randint(100, 500)  # base traffic volume per location
        daily_pattern = (np.sin(np.arange(hours) * (2 * np.pi / 24)) + 1) * 0.5  # peak during day hours
        weekly_pattern = (np.cos(np.arange(hours) * (2 * np.pi / (24*7))) + 1) * 0.5  # weekend lower traffic
        noise = np.random.normal(0, 0.1, hours)
        traffic_volume = base_volume * (0.5 + daily_pattern + 0.5 * weekly_pattern) + noise * base_volume
        traffic_volume = np.clip(traffic_volume, 0, None)  # no negative traffic
        timestamps = pd.date_range("2023-01-01", periods=hours, freq='H')
        for ts, vol in zip(timestamps, traffic_volume):
            data.append({'location': f'Location_{loc+1}', 'timestamp': ts, 'traffic_volume': vol})
    df = pd.DataFrame(data)
    return df

def create_features(df):
    """
    Create time-based features for ML model from timestamp.
    """
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    # Lag features to capture recent traffic volume trends
    df['lag_1'] = df.groupby('location')['traffic_volume'].shift(1)
    df['lag_2'] = df.groupby('location')['traffic_volume'].shift(2)
    df['lag_3'] = df.groupby('location')['traffic_volume'].shift(3)
    df = df.dropna()
    return df

def train_predictive_model(df):
    """
    Train a RandomForest regressor to predict traffic volume based on features.
    """
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['traffic_volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error on test set: {mse:.2f}")
    return model

def predict_future_traffic(df, model, periods=24):
    """
    Predict future traffic volumes for each location for the next 'periods' hours.
    """
    future_data = []
    locations = df['location'].unique()
    last_timestamp = df['timestamp'].max()
    for loc in locations:
        last_rows = df[df['location'] == loc].sort_values('timestamp').iloc[-3:].copy()
        for i in range(1, periods+1):
            future_ts = last_timestamp + pd.Timedelta(hours=i)
            hour = future_ts.hour
            dayofweek = future_ts.dayofweek
            day = future_ts.day
            month = future_ts.month
            lag_1 = last_rows.iloc[-1]['traffic_volume']
            lag_2 = last_rows.iloc[-2]['traffic_volume']
            lag_3 = last_rows.iloc[-3]['traffic_volume']
            features = pd.DataFrame({
                'hour': [hour],
                'dayofweek': [dayofweek],
                'day': [day],
                'month': [month],
                'lag_1': [lag_1],
                'lag_2': [lag_2],
                'lag_3': [lag_3]
            })
            pred_vol = model.predict(features)[0]
            future_data.append({'location': loc, 'timestamp': future_ts, 'predicted_traffic_volume': pred_vol})
            # Update last_rows for next iteration
            new_row = {'traffic_volume': pred_vol}
            last_rows = last_rows.append(pd.Series(new_row), ignore_index=True).iloc[1:]
    future_df = pd.DataFrame(future_data)
    return future_df

def detect_congestion_hotspots(pred_df, threshold=600):
    """
    Detect congestion hotspots where predicted traffic volume exceeds a threshold.
    """
    hotspots = pred_df[pred_df['predicted_traffic_volume'] > threshold]
    print(f"Detected {len(hotspots)} congestion hotspot predictions (threshold: {threshold})")
    return hotspots

def plot_traffic_trends(df, pred_df, location):
    """
    Plot historical and predicted traffic volumes for a given location.
    """
    plt.figure(figsize=(12,6))
    hist_data = df[df['location'] == location].set_index('timestamp')
    pred_data = pred_df[pred_df['location'] == location].set_index('timestamp')
    plt.plot(hist_data.index, hist_data['traffic_volume'], label='Historical Traffic')
    plt.plot(pred_data.index, pred_data['predicted_traffic_volume'], label='Predicted Traffic', linestyle='--')
    plt.title(f"Traffic Volume Trends for {location}")
    plt.xlabel("Time")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Generating synthetic traffic data...")
    df = generate_synthetic_traffic_data(num_locations=5, days=30)
    print("Creating features for model training...")
    df_feat = create_features(df)
    print("Training traffic volume prediction model...")
    model = train_predictive_model(df_feat)
    print("Predicting future traffic volumes for next 24 hours...")
    pred_df = predict_future_traffic(df_feat, model, periods=24)
    print("Detecting congestion hotspots...")
    hotspots = detect_congestion_hotspots(pred_df, threshold=600)
    print(hotspots)
    # Plot traffic trends for one location as example
    plot_traffic_trends(df, pred_df, location='Location_1')



