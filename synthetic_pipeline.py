import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
def generate_synthetic_traffic_data(num_locations=5, days=30):
    np.random.seed(42)
    rows = []
    for loc in range(num_locations):
        for day in range(days):
            for hour in range(24):
                base = np.random.randint(100, 500)
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    vol = base + np.random.randint(100, 300)
                else:
                    vol = base + np.random.randint(0, 100)
                rows.append({'location_id': loc, 'day': day, 'hour': hour, 'traffic_volume': vol})
    return pd.DataFrame(rows)

# 2. Create features
def create_features(df):
    df = df.copy()
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['lag_1'] = df['traffic_volume'].shift(1).fillna(method='bfill')
    df['lag_2'] = df['traffic_volume'].shift(2).fillna(method='bfill')
    df['lag_3'] = df['traffic_volume'].shift(3).fillna(method='bfill')
    return df.dropna()

# 3. Train model
def train_predictive_model(df):
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    X, y = df[features], df['traffic_volume']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)
    return model

# Pipeline entrypoint
if __name__ == "__main__":
    df = generate_synthetic_traffic_data(num_locations=3, days=14)
    df_feat = create_features(df)
    model = train_predictive_model(df_feat)
    df_feat['predicted_traffic_volume'] = model.predict(df_feat[['hour','dayofweek','day','month','lag_1','lag_2','lag_3']])
    df_feat['true_category'] = df_feat['traffic_volume'].apply(lambda v: 'Low' if v < 200 else 'Medium' if v < 400 else 'High')
    df_feat['predicted_category'] = df_feat['predicted_traffic_volume'].apply(lambda v: 'Low' if v < 200 else 'Medium' if v < 400 else 'High')
    df_feat.to_csv('synthetic_predictions.csv', index=False)
    print("Synthetic pipeline complete. Predictions saved to 'synthetic_predictions.csv'")