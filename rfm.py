# Calculate Recency, Frequency, and Monetary values
import datetime as dt
from sklearn.preprocessing import StandardScaler 


def calculate_rfm(data):
    # Set the reference date for Recency calculation
    reference_date = dt.datetime(2011, 12, 31)

    # Calculate Recency (days since last purchase)
    data['Recency'] = (reference_date - data['InvoiceDate']).dt.days
    rfm = data.groupby('CustomerID').agg({
        'Recency': 'min',  # Minimum recency
        'InvoiceNo': 'count',  # Frequency
        'Amount': 'sum'  # Monetary value
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    #remove the negative monetary values
    rfm = rfm[rfm['Monetary'] >= 0]

    return rfm

def normalize_data(data):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary']])

    print(rfm_scaled[:5])
    return rfm_scaled
