import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def synthetic_traffic_data(num_locations=5, days=30):
    locations = [f'Location_{i+1}' for i in range(num_locations)]
    start_time = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, periods=days * 24, freq='H')

    data = []
    for location in locations:
        volume = np.random.randint(100, 800, size=len(timestamps)) + \
                 np.sin(np.linspace(0, 3*np.pi, len(timestamps))) * 100  # simulate daily pattern
        volume = np.maximum(0, volume).astype(int)
        for i, time in enumerate(timestamps):
            data.append({'timestamp': time, 'location': location, 'traffic_volume': volume[i]})

    return pd.DataFrame(data)
