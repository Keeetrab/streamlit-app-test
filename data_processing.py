import pandas as pd


def clean_data(data):
    # Convert 'UnitPrice' to numeric
    data['UnitPrice'] = data['UnitPrice'].str.replace(',', '.')
    data['UnitPrice'] = pd.to_numeric(data['UnitPrice'])

    # Create 'Amount' column
    data['Amount'] = data['Quantity'] * data['UnitPrice']

    # Separate 'InvoiceDate' into 'Date' and 'Time'
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['Date'] = data['InvoiceDate'].dt.date
    data['Time'] = data['InvoiceDate'].dt.time

    # Filter transactions by UK postcodes
    data = data[data['Country'] == 'United Kingdom']


    return data