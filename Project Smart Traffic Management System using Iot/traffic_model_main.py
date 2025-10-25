import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_synthetic_traffic_data(num_locations=5, days=30):
    np.random.seed(42)
    data = []
    for loc in range(num_locations):
        for day in range(days):
            for hour in range(24):
                base_volume = np.random.randint(100, 500)
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    volume = base_volume + np.random.randint(100, 300)
                else:
                    volume = base_volume + np.random.randint(0, 100)
                data.append({
                    'location_id': loc,
                    'day': day,
                    'hour': hour,
                    'traffic_volume': volume
                })
    df = pd.DataFrame(data)
    return df

def create_features(df):
    df['timestamp'] = pd.date_range(start='2025-01-01', periods=len(df), freq='H')
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['lag_1'] = df['traffic_volume'].shift(1).fillna(method='bfill')
    df['lag_2'] = df['traffic_volume'].shift(2).fillna(method='bfill')
    df['lag_3'] = df['traffic_volume'].shift(3).fillna(method='bfill')
    return df.dropna()

def train_predictive_model(df):
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['traffic_volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def categorize_traffic_volume(volume):
    if volume < 200:
        return 'Low'
    elif 200 <= volume < 400:
        return 'Medium'
    else:
        return 'High'

if __name__ == "__main__":
    # Generate data and create features
    df = generate_synthetic_traffic_data(num_locations=3, days=14)
    df_feat = create_features(df)

    # Train model
    model = train_predictive_model(df_feat)

    # Predict on all data for evaluation
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    y_true = df_feat['traffic_volume']
    y_pred = model.predict(df_feat[features])

    # Save true and predicted values and their categories to CSV for later evaluation
    results_df = df_feat.copy()
    results_df['predicted_traffic_volume'] = y_pred
    results_df['true_category'] = y_true.apply(categorize_traffic_volume)
    results_df['predicted_category'] = results_df['predicted_traffic_volume'].apply(categorize_traffic_volume)

    results_df.to_csv('traffic_predictions.csv', index=False)
    print("Saved predictions and true labels to 'traffic_predictions.csv'")