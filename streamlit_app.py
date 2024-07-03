import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from anomalies import filter_anomalies_with_isolation_forest
from clustering import calculate_cluster_statistics, elbow_method, monetary_pie_chart, plot_clusters, transaction_pie_chart
from data_processing import clean_data
from rfm import calculate_rfm, normalize_data


# Show the page title and description.
st.set_page_config(page_title="Retail Segmentation test", page_icon="🎬")
st.title("Retail Segmentation Case")
st.write(
    """
    Sample Retail Segmentation Case using database from UC Irvine Machine Learning Repository
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    data = pd.read_csv("my_data/retail_data.csv")
    return data

#___Getting Data____
data = load_data()

st.subheader("Data Preview")
st.write(data.head())
st.write(data.describe())

#___Preprocessing___
data = clean_data(data)

st.subheader("Preprocessing")
st.write("Cleaned data by: swaping , to . in UnitPrices, spltiing Invoice Data into Date and Time, added Amount column (Quantity * UnityPrice), filtered country to UK only ")
st.write(data)


#___RFM___

rfm = calculate_rfm(data)
st.subheader("RFM")
st.write("Calculate Recency, Frequency and Monetary Values")
st.write(rfm)



# _____K-means clustering_____
st.subheader("Clustering")
st.write("For clustering I use K-means clustering with Elbow Method")
#Filter out anomalies
rfm_cleaned = filter_anomalies_with_isolation_forest(rfm)

# Estimtate number of clusters
st.write("Elbow Method")
plt = elbow_method(rfm_cleaned)
st.pyplot(plt)

optimal_clusters = 4

# Normalize Data
rfm_cleaned_normalized = normalize_data(rfm_cleaned)

# K-means clustering
st.write("K-means Clustering")
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
rfm_cleaned['Cluster'] = kmeans.fit_predict(rfm_cleaned_normalized)

st.write(rfm_cleaned.head())

#cluster statistics
cluster_analysis = calculate_cluster_statistics(rfm_cleaned)
st.write("Cluster Statistics")
st.dataframe(cluster_analysis)

# Plot the 3D scatter plots using Matplotlib
fig = plot_clusters(rfm_cleaned)
st.write("Cluster Graphs")
st.pyplot(fig)

#pie charts

st.pyplot(transaction_pie_chart(rfm_cleaned))
st.pyplot(monetary_pie_chart(rfm_cleaned))

st.subheader("Cluster Analysis and Interpretation")
st.write('''
Cluster 0: Inactive / Low-value

Characteristics: High Recency (mean: 265.6), low Frequency (mean: 26.7), low Monetary value (mean: 395,6).
Interpretation: These customers have not purchased recently, buy infrequently, and spend less money. They could be categorized as "inactive" or "low-value" customers.
Strategy: Reactivate these customers through targeted promotions and discounts.
         

Cluster 1: Active Customers
         
Characteristics: Moderate Recency (mean: 65.5), moderate Frequency (mean: 58.8), moderate Monetary value (mean: 976,3).
Interpretation: These customers purchase relatively frequently and have a decent spending amount. They can be seen as "active" customers.
Strategy: Engage these customers with loyalty programs and personalized offers to maintain and increase their activity.

         
Cluster 2: Very High Value / Super VIP / Enterprise Clients

Characteristics: Low Recency (mean: 32.9), high Frequency (mean: 264.6), very high Monetary value (mean: 14633,1).
Interpretation: These customers are recent, frequent, and have very high spending. They are the most valuable customers, possibly "super VIPs".
Strategy: Offer top-tier loyalty programs, personalized experiences, and high-touch customer service to ensure their continued loyalty.


Cluster 3: High Value / VIP Customers / Small Business Clients
         
Characteristics: Low Recency (mean: 39.9), very high Frequency (mean: 325.1), high Monetary value (mean: 4005.5).
Interpretation: These are recent, frequent buyers with high spending. They are likely the "high-value" or "Small Business" customers.
Strategy: Provide exclusive benefits, early access to new products, and premium services to retain these valuable customers.

         
''')



# ______Training Model________

st.title("Model training")
# Split data into features and target
X = rfm_cleaned[['Recency', 'Frequency', 'Monetary']]
y = rfm_cleaned['Cluster']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Learning rate
    max_depth=3,  # Maximum depth of a tree
    subsample=0.8,  # Subsample ratio of the training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
    random_state=42  # Random seed
)
xgb_model.fit(X_train, y_train)
# Training
y_train_pred = xgb_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
st.write(f'Accuracy Training Set: {train_accuracy}')


# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the results in Streamlit
st.write(f'Accuracy Test Set: {accuracy}')
st.write('Confusion Matrix:')
st.write(conf_matrix)
st.write('Classification Report:')
st.write(class_report)

# Feature importance plot
fig, ax = plt.subplots()
xgb.plot_importance(xgb_model, ax=ax)
st.pyplot(fig)