import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_features(df, timestamp_col='timestamp'):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek
    df['day'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['lag_1'] = df.groupby('location')['traffic_volume'].shift(1)
    df['lag_2'] = df.groupby('location')['traffic_volume'].shift(2)
    df['lag_3'] = df.groupby('location')['traffic_volume'].shift(3)
    return df.dropna().reset_index(drop=True)

def train_predictive_model(df):
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    X = df[features]
    y = df['traffic_volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    return model

def predict_future_traffic(df, model, periods=24):
    results = []
    features = ['hour', 'dayofweek', 'day', 'month', 'lag_1', 'lag_2', 'lag_3']
    for loc in df['location'].unique():
        subset = df[df['location'] == loc].sort_values('timestamp')
        last_rows = subset.iloc[-3:].copy()
        last_ts = subset['timestamp'].max()
        for i in range(1, periods + 1):
            ts = last_ts + pd.Timedelta(hours=i)
            hour, dow, day, month = ts.hour, ts.dayofweek, ts.day, ts.month
            lag1, lag2, lag3 = (
                last_rows['traffic_volume'].iloc[-1],
                last_rows['traffic_volume'].iloc[-2],
                last_rows['traffic_volume'].iloc[-3],
            )
            Xf = pd.DataFrame([{
                'hour': hour, 'dayofweek': dow, 'day': day, 'month': month,
                'lag_1': lag1, 'lag_2': lag2, 'lag_3': lag3
            }])
            pred = model.predict(Xf)[0]
            results.append({'location': loc, 'timestamp': ts, 'predicted_traffic': pred})
            new_row = pd.DataFrame([{'traffic_volume': pred}])
            last_rows = pd.concat([last_rows, new_row], ignore_index=True).iloc[1:]
    return pd.DataFrame(results)
