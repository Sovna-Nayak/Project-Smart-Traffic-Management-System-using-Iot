import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------
# Feature Engineering Module
# -------------------------

def create_features(df, timestamp_col: str = None):
    """
    Add time-based and lag features.
    - If timestamp_col is provided, use that.
    - Otherwise, auto-detect a datetime column among common names.
    """
    df = df.copy()

    # Determine timestamp column
    if timestamp_col and timestamp_col in df.columns:
        ts = timestamp_col
    else:
        candidates = ['timestamp', 'time', 'date', 'datetime', 'DateTime', 'Date']
        ts = next((c for c in candidates if c in df.columns), None)
        if ts is None:
            print("Chicago CSV columns:", df.columns.tolist())
            raise KeyError(f"No timestamp column found. Tried: {candidates}")

    # Convert and extract features
    df[ts] = pd.to_datetime(df[ts])
    df['hour'] = df[ts].dt.hour
    df['dayofweek'] = df[ts].dt.dayofweek
    df['day'] = df[ts].dt.day
    df['month'] = df[ts].dt.month

    # Lag features per location
    df['lag_1'] = df.groupby('location')['traffic_volume'].shift(1)
    df['lag_2'] = df.groupby('location')['traffic_volume'].shift(2)
    df['lag_3'] = df.groupby('location')['traffic_volume'].shift(3)

    return df.dropna().reset_index(drop=True)

# -------------------------
# Model Training & Prediction Module
# -------------------------

def train_predictive_model(df):
    """
    Train a RandomForest regressor to predict traffic volume based on features.
    """
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['traffic_volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error on test set: {mse:.2f}")

    return model


def predict_future_traffic(df, model, periods=24):
    """
    Predict future traffic volumes for each location for the next `periods` hours.
    """
    results = []
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']

    for loc in df['location'].unique():
        subset = df[df['location'] == loc].sort_values('timestamp')
        last_rows = subset.iloc[-3:].copy()
        last_ts = subset['timestamp'].max()

        for i in range(1, periods + 1):
            ts = last_ts + pd.Timedelta(hours=i)
            hour = ts.hour
            dow = ts.dayofweek
            day = ts.day
            month = ts.month

            lag1, lag2, lag3 = (
                last_rows['traffic_volume'].iloc[-1],
                last_rows['traffic_volume'].iloc[-2],
                last_rows['traffic_volume'].iloc[-3]
            )

            Xf = pd.DataFrame([{ 
                'hour': hour,
                'dayofweek': dow,
                'day': day,
                'month': month,
                'lag_1': lag1,
                'lag_2': lag2,
                'lag_3': lag3
            }])

            pred = model.predict(Xf)[0]
            results.append({
                'location': loc,
                'timestamp': ts,
                'predicted_traffic': pred
            })

            # Update lag window
            new_row = pd.DataFrame([{'traffic_volume': pred}])
            last_rows = pd.concat([last_rows, new_row], ignore_index=True).iloc[1:]

    return pd.DataFrame(results)

# -------------------------
# Data Generation Module
# -------------------------

def generate_synthetic_traffic_data(num_locations=5, days=30):
    """
    Generate synthetic hourly traffic volume data for multiple locations.
    """
    np.random.seed(42)
    hours = days * 24
    data = []

    for loc in range(num_locations):
        base_volume = np.random.randint(100, 500)
        daily_pattern = (np.sin(np.arange(hours) * (2 * np.pi / 24)) + 1) * 0.5
        weekly_pattern = (np.cos(np.arange(hours) * (2 * np.pi / (24*7))) + 1) * 0.5
        noise = np.random.normal(0, 0.1, hours)

        traffic_volume = (
            base_volume * (0.5 + daily_pattern + 0.5 * weekly_pattern)
            + noise * base_volume
        )
        traffic_volume = np.clip(traffic_volume, 0, None)
        timestamps = pd.date_range("2023-01-01", periods=hours, freq='H')

        for ts, vol in zip(timestamps, traffic_volume):
            data.append({
                'location': f'Location_{loc+1}',
                'timestamp': ts,
                'traffic_volume': vol
            })

    return pd.DataFrame(data)

# -------------------------
# Pipeline Entry Point
# -------------------------

if __name__ == "__main__":
    # 1. Synthetic Data
    df_synth = generate_synthetic_traffic_data(num_locations=3, days=14)
    print("Generated synthetic data sample:")
    print(df_synth.head(), "\n")
    df_synth.to_csv("synthetic_traffic_data.csv", index=False)

    # 2. Feature Engineering (Synthetic)
    df_feat = create_features(df_synth)
    print("Feature-engineered data sample:")
    print(df_feat.head(), "\n")
    df_feat.to_csv("feature_engineered_traffic_data.csv", index=False)

    # 3. Train Model
    model = train_predictive_model(df_feat)

    # 4. Predict Future (Synthetic)
    df_future = predict_future_traffic(df_feat, model, periods=24)
    print("Future traffic predictions (synthetic) sample:")
    print(df_future.head(10), "\n")
    df_future.to_csv("future_traffic_predictions.csv", index=False)

    # 5. Load & Process Chicago Data
    try:
        df_chi = pd.read_csv("chicago_traffic.csv")
        # Option: Uncomment and rename if your time column differs
        # df_chi = df_chi.rename(columns={'YOUR_DATE_COLUMN': 'timestamp'})
        df_chi_feat = create_features(df_chi, timestamp_col='timestamp')
        df_chi_future = predict_future_traffic(df_chi_feat, model, periods=24)
        df_chi_future.to_csv("chicago_traffic_forecast.csv", index=False)
        print("Chicago traffic forecast saved to 'chicago_traffic_forecast.csv'\n")
    except Exception as e:
        print(f"Warning: Could not process Chicago data: {e}\n")

    # 6. Plot Example
    #loc = "Location_1"
    #sub = df_future[df_future['location'] == loc]
    #plt.figure(figsize=(10,4))
    #plt.plot(sub['timestamp'], sub['predicted_traffic'], marker='o')
    #plt.title(f"24h Traffic Forecast for {loc}")
    #plt.xlabel("Timestamp")
    #plt.ylabel("Predicted Traffic Volume")
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #png = f"traffic_forecast_{loc}.png"
    #plt.savefig(png)
    #print(f"Saved plot to '{png}'")
    #plt.show()
    #also draw the bar chart
        # 6. Plot Example as Bar Chart
    loc = "Location_1"
    sub = df_future[df_future['location'] == loc]
    plt.figure(figsize=(12, 5))
    plt.bar(sub['timestamp'].astype(str), sub['predicted_traffic'], color='skyblue')
    plt.title(f"24h Traffic Forecast for {loc} (Bar Chart)")
    plt.xlabel("Timestamp")
    plt.ylabel("Predicted Traffic Volume")
    plt.xticks(rotation=90)
    plt.tight_layout()
    png = f"traffic_forecast_bar_{loc}.png"
    plt.savefig(png)
    print(f"Saved bar chart to '{png}'")
    plt.show()

