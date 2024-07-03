from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest


def filter_anomalies_with_isolation_forest(data):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary']])
    # Detect anomalies using Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    anomalies = iso_forest.fit_predict(data_scaled)
    data['Anomaly'] = anomalies

    # Filter out anomalies
    data_cleaned = data[data['Anomaly'] != -1]
    return data_cleaned